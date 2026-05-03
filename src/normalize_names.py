import io
import json
import csv
import argparse
import re
import sys
import threading
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import ollama as _ollama
except ImportError:
    print("Run: pip install ollama")
    sys.exit(1)

DEFAULT_MODEL = "llama3.1:8b"
BATCH_SIZE = 20

NORMALIZE_PROMPT = (
    "Ти аналізуєш список імен людей з архівних документів КГБ/ОГПУ 1920–1930-х років.\n\n"
    "Одна людина може зустрічатися по-різному:\n"
    "  — різні відмінки (слов'янські): «КОТЛЯРА», «Котляру», «Котляр»\n"
    "  — відмінювання неслов'янських прізвищ за рос. граматикою: «ХАТТАТОВЫМ» → «Хаттатов»,\n"
    "    «ГОВЕНОМ» → «Говен», «ГАЛИЕВЫХ» → «Галієв», «БЕЛА-КУНА» → «Бела-Кун»\n"
    "  — різна повнота: «Коваленко», «Коваленко П.», «Коваленко Петро», «Коваленко Петро Семенович»\n"
    "  — укр/рос написання: «Сітдіков» і «Ситдиков»\n"
    "  — CAPS і TitleCase: «БРОНШТЕЙНА» і «Бронштейн»\n"
    "  — чоловіча/жіноча форма прізвища: «Сітдіков» і «Сітдікова» — та сама людина\n\n"
    "Правила:\n"
    "  1. Якщо одне ім'я є скороченням іншого (однакове прізвище) — це та сама людина. "
    "Об'єднуй: «Коваленко» + «Коваленко П.» + «Коваленко Петро» → одна група.\n"
    "     НЕ об'єднуй людей з різними прізвищами: «Петренко» і «Гриценко» — різні люди.\n"
    "  2. Канонічна форма — НАЙПОВНІША з наявних варіантів, Title Case, називний відмінок.\n"
    "     Пріоритет: Прізвище Ім'я По-батькові > Прізвище Ім'я > Прізвище Ініціал. > Прізвище.\n"
    "     Для визначення статі (Сітдіков vs Сітдікова) використовуй приклади з тексту.\n"
    "  3. Кожне ім'я зі списку — рівно в одній групі.\n\n"
    "Поверни JSON-об'єкт з полем \"groups\":\n"
    "{\"groups\": [{\"canonical\": \"Котляр Григорій Кузьмич\", "
    "\"variants\": [\"КОТЛЯРА\", \"Котляр Г.К.\", \"Котляр\", \"Котляр Григорій Кузьмич\"]}, ...]}"
)


def _stem4(name: str) -> str:
    first = name.strip().split()[0] if name.strip() else name
    return re.sub(r"[^а-яёіїєА-ЯЁІЇЄa-z]", "", first).lower()[:4]


def _words(name: str) -> list[str]:
    return re.split(r"[\s\-]+", name.lower())


def _stems_match(a: str, b: str) -> bool:
    """Used to validate LLM merges: allows 1-char transliteration diff (stems >= 4 chars)."""
    if a.lower() == b.lower():
        return True
    aw, bw = _words(a), _words(b)

    # Multi-word same count: all word-positions must match (reuse _strict logic)
    if len(aw) > 1 and len(aw) == len(bw):
        for w1, w2 in zip(aw, bw):
            if w1 == w2:
                continue
            s1, s2 = w1[:4], w2[:4]
            if len(s1) < 4 or len(s2) < 4 or s1[0] != s2[0]:
                return False
            if sum(c1 != c2 for c1, c2 in zip(s1, s2)) > 1:
                return False
        return True

    # Single-word or different lengths: original stem4 logic
    s1, s2 = _stem4(a), _stem4(b)
    if s1 == s2:
        return True
    if len(s1) >= 4 and len(s2) >= 4:
        if sum(c1 != c2 for c1, c2 in zip(s1, s2)) <= 1:
            return True
    # word prefix/suffix: "Муслюмов" ⊂ "Аким Муслюмов"; "Чобан-Заде" ⊂ "Чобан Заде Бекир"
    if aw and bw and len(aw) != len(bw):
        shorter, longer = (aw, bw) if len(aw) < len(bw) else (bw, aw)
        if longer[:len(shorter)] == shorter or longer[-len(shorter):] == shorter:
            return True
    return False


def _merge_by_stem(names: list[str], freq: dict[str, int] | None = None) -> dict[str, str]:
    """Deterministic post-pass: merge transliteration variants and short-form names.

    Rules:
    - case-insensitive exact match → merge
    - same word count: all word-positions must match (exact OR 1-char stem4 diff, len>=4)
      single-word exception: skip if one string is a prefix of the other (Сейдамет/Сейдаметов)
    - different word counts: word prefix/suffix OR joined-string match (Чобанзаде/Чобан-Заде)
    """
    def _strict(a: str, b: str) -> bool:
        if a.lower() == b.lower():
            return True
        aw, bw = _words(a), _words(b)

        if len(aw) != len(bw):
            shorter, longer = (aw, bw) if len(aw) < len(bw) else (bw, aw)
            if longer[:len(shorter)] == shorter or longer[-len(shorter):] == shorter:
                return True
            # "Чобанзаде" ≡ "Чобан Заде ..." — joined shorter is prefix/suffix of joined longer
            js, jl = "".join(shorter), "".join(longer)
            return jl.startswith(js) or jl.endswith(js)

        # same word count: all positions must match within 1-char stem diff
        for w1, w2 in zip(aw, bw):
            if w1 == w2:
                continue
            s1, s2 = w1[:4], w2[:4]
            if len(s1) < 4 or len(s2) < 4:
                return False
            # single-word: if one extends the other, only allow oblique case endings
            if len(aw) == 1 and (w1.startswith(w2) or w2.startswith(w1)):
                longer_w = w1 if len(w1) > len(w2) else w2
                if not any(longer_w.endswith(e) for e in _OBLIQUE_ENDS):
                    return False
            # first char must match — different initial consonants = different people
            if s1[0] != s2[0]:
                return False
            if sum(c1 != c2 for c1, c2 in zip(s1, s2)) > 1:
                return False
        return True

    ordered = sorted(names, key=lambda n: (-(freq or {}).get(n, 0), -len(n)))
    groups: list[list[str]] = []
    for name in ordered:
        placed = False
        for g in groups:
            if any(_strict(name, m) for m in g):
                g.append(name)
                placed = True
                break
        if not placed:
            groups.append([name])
    return {name: _best_canonical(g, freq) for g in groups for name in g}


def _to_title(name: str) -> str:
    return " ".join(w.title() if w == w.upper() and len(w) > 1 else w for w in name.split())


def _extract_json(raw: str) -> list:
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return []


def normalize_batch(names: list[str], model: str,
                    context: dict[str, str] | None = None,
                    freq: dict[str, int] | None = None) -> dict[str, str]:
    lines = []
    for n in names:
        ctx = context.get(n, "") if context else ""
        lines.append(f"- {n}" + (f"  (приклад з тексту: «{ctx}»)" if ctx else ""))
    names_text = "\n".join(lines)
    stop = threading.Event()
    parts: list[str] = []
    exc: list = [None]

    def _call():
        try:
            for chunk in _ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": NORMALIZE_PROMPT},
                    {"role": "user", "content": names_text},
                ],
                format="json",
                options={"temperature": 0.0, "num_ctx": 4096, "num_predict": 512, "num_thread": 6},
                stream=True,
            ):
                if stop.is_set():
                    break
                parts.append(chunk.message.content or "")
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=180)
    if t.is_alive():
        stop.set()
        print("  [WARN] LLM timeout, skipping batch")
        return {n: n for n in names}
    if exc[0]:
        print(f"  [WARN] LLM error: {exc[0]}")
        return {n: n for n in names}
    groups = _extract_json("".join(parts))

    mapping: dict[str, str] = {}
    seen = set()
    for group in groups:
        if not isinstance(group, dict):
            continue
        canonical = group.get("canonical", "")
        if not canonical:
            continue
        for variant in group.get("variants", []):
            if isinstance(variant, str):
                mapping[variant] = canonical
                seen.add(variant)
        mapping[canonical] = canonical
        seen.add(canonical)

    for name in names:
        if name not in seen:
            mapping[name] = name

    names_set = set(names)
    validated: dict[str, str] = {}
    for k, v in mapping.items():
        if k not in names_set:
            continue
        if k == v:
            validated[k] = _to_title(k)
        elif _stems_match(k, v):
            if v in names_set:
                validated[k] = _to_title(v)
            else:
                same_stem = [n for n in names_set if _stems_match(n, v)]
                validated[k] = _best_canonical(same_stem, freq) if same_stem else _to_title(k)
        else:
            validated[k] = _to_title(k)
    return validated


_OBLIQUE_ENDS = ("ым", "им", "ых", "их", "ом", "ем", "ого", "его", "ому", "ему")


def _best_canonical(candidates: list[str], freq: dict[str, int] | None = None) -> str:
    """Prefer nominative over oblique/genitive, then longest (fuller name), then most frequent."""
    def score(n: str) -> tuple:
        low = n.lower()
        oblique = any(low.endswith(e) for e in _OBLIQUE_ENDS)
        gen = low[-1] in ("а", "я") and not low.endswith(("ко", "нко")) if low else False
        frequency = freq.get(n, 0) if freq else 0
        return (not oblique, not gen, len(n), frequency)
    return _to_title(max(candidates, key=score))


def run(input_path: str | Path, output_dir: str | Path, model: str,
        keep_raw: bool = False) -> tuple[Path, Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8", newline="") as f:
        edges_raw = list(csv.DictReader(f))

    all_names = sorted({
        name.strip()
        for edge in edges_raw
        for name in (edge["source"], edge["target"])
        if name.strip()
    })

    print(f"Unique raw names: {len(all_names)}")
    for n in all_names:
        print(f"  {n}")

    name_context: dict[str, str] = {}
    name_freq: dict[str, int] = {}
    for edge in edges_raw:
        for field in ("source", "target"):
            name = edge[field].strip()
            if not name:
                continue
            name_freq[name] = name_freq.get(name, 0) + 1
            if name not in name_context and edge.get("evidence_quote"):
                name_context[name] = edge["evidence_quote"][:80]

    full_mapping: dict[str, str] = {}

    batches = [all_names[i: i + BATCH_SIZE] for i in range(0, len(all_names), BATCH_SIZE)]
    for idx, batch in enumerate(batches, 1):
        print(f"Normalizing batch {idx}/{len(batches)} ({len(batch)} names) ...", flush=True)
        batch_map = normalize_batch(batch, model, context=name_context, freq=name_freq)
        full_mapping.update(batch_map)
        for raw, canonical in sorted(batch_map.items()):
            if raw != canonical:
                print(f"  {raw!r} → {canonical!r}")

    canonical_freq: dict[str, int] = {}
    for raw, canon in full_mapping.items():
        canonical_freq[canon] = canonical_freq.get(canon, 0) + name_freq.get(raw, 0)

    canonicals = list(set(full_mapping.values()))
    stem_map = _merge_by_stem(canonicals, canonical_freq)
    extra_merges = {k: v for k, v in stem_map.items() if k != v}
    if extra_merges:
        print(f"\nStem post-pass ({len(extra_merges)} merges):")
        for k, v in sorted(extra_merges.items()):
            print(f"  {k!r} → {v!r}")
    for k in full_mapping:
        full_mapping[k] = stem_map.get(full_mapping[k], full_mapping[k])

    edges_final = []
    for edge in edges_raw:
        src = full_mapping.get(edge["source"], edge["source"])
        tgt = full_mapping.get(edge["target"], edge["target"])
        if src.lower() == tgt.lower():
            continue
        edges_final.append({
            "source": src,
            "target": tgt,
            "sentiment": edge["sentiment"],
            "evidence_quote": edge["evidence_quote"],
            "id": edge["id"],
        })

    node_ids = sorted({e["source"] for e in edges_final} | {e["target"] for e in edges_final})

    nodes_path = output_dir / "nodes_ollama.csv"
    edges_path = output_dir / "edges_ollama.csv"

    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label"])
        w.writeheader()
        for nid in node_ids:
            w.writerow({"id": nid, "label": nid})

    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["source", "target", "sentiment", "evidence_quote", "id"],
        )
        w.writeheader()
        w.writerows(edges_final)

    if not keep_raw:
        input_path.unlink(missing_ok=True)

    print(f"\nNodes: {len(node_ids)}, Edges: {len(edges_final)}")
    print(f"Saved: {nodes_path}")
    print(f"Saved: {edges_path}")
    return nodes_path, edges_path


def main():
    parser = argparse.ArgumentParser(description="KGB name normalizer — pass 2")
    parser.add_argument("--input", default="data/processed/edges_raw_ollama.csv")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--keep-raw", action="store_true",
                        help="Keep edges_raw.csv after normalization")
    args = parser.parse_args()

    try:
        _ollama.list()
    except Exception as e:
        print(f"[ERROR] Ollama not reachable: {e}")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] {input_path} not found. Run extract_entities.py first.")
        sys.exit(1)

    run(input_path, args.output_dir, args.model, keep_raw=args.keep_raw)


if __name__ == "__main__":
    main()
