import io
import json
import csv
import argparse
import re
import sys
import os
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import anthropic as _anthropic
except ImportError:
    print("Run: pip install anthropic")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 50

_COST_IN  = 0.80 / 1_000_000
_COST_OUT = 4.00 / 1_000_000

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
    """Validates LLM merges: allows 1-char transliteration diff (stems >= 4 chars)."""
    if a.lower() == b.lower():
        return True
    aw, bw = _words(a), _words(b)

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

    s1, s2 = _stem4(a), _stem4(b)
    if s1 == s2:
        return True
    if len(s1) >= 4 and len(s2) >= 4:
        if sum(c1 != c2 for c1, c2 in zip(s1, s2)) <= 1:
            return True
    if aw and bw and len(aw) != len(bw):
        shorter, longer = (aw, bw) if len(aw) < len(bw) else (bw, aw)
        if longer[:len(shorter)] == shorter or longer[-len(shorter):] == shorter:
            return True
    return False


_OBLIQUE_ENDS = ("ым", "им", "ых", "их", "ом", "ем", "ого", "его", "ому", "ему")


def _best_canonical(candidates: list[str], freq: dict[str, int] | None = None) -> str:
    def score(n: str) -> tuple:
        low = n.lower()
        oblique = any(low.endswith(e) for e in _OBLIQUE_ENDS)
        gen = low[-1] in ("а", "я") and not low.endswith(("ко", "нко")) if low else False
        frequency = freq.get(n, 0) if freq else 0
        return (not oblique, not gen, len(n), frequency)
    return _to_title(max(candidates, key=score))


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


def _merge_by_stem(names: list[str], freq: dict[str, int] | None = None) -> dict[str, str]:
    def _strict(a: str, b: str) -> bool:
        if a.lower() == b.lower():
            return True
        aw, bw = _words(a), _words(b)

        if len(aw) != len(bw):
            shorter, longer = (aw, bw) if len(aw) < len(bw) else (bw, aw)
            if longer[:len(shorter)] == shorter or longer[-len(shorter):] == shorter:
                return True
            js, jl = "".join(shorter), "".join(longer)
            return jl.startswith(js) or jl.endswith(js)

        for w1, w2 in zip(aw, bw):
            if w1 == w2:
                continue
            s1, s2 = w1[:4], w2[:4]
            if len(s1) < 4 or len(s2) < 4:
                return False
            if len(aw) == 1 and (w1.startswith(w2) or w2.startswith(w1)):
                longer_w = w1 if len(w1) > len(w2) else w2
                if not any(longer_w.endswith(e) for e in _OBLIQUE_ENDS):
                    return False
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


def normalize_batch(names: list[str], client: "_anthropic.Anthropic", model: str,
                    context: dict[str, str] | None = None,
                    freq: dict[str, int] | None = None,
                    usage: dict | None = None) -> dict[str, str]:
    lines = []
    for n in names:
        ctx = context.get(n, "") if context else ""
        lines.append(f"- {n}" + (f"  (приклад з тексту: «{ctx}»)" if ctx else ""))
    names_text = "\n".join(lines)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=NORMALIZE_PROMPT,
            messages=[{"role": "user", "content": names_text}],
        )
        if usage is not None:
            usage["in"] += response.usage.input_tokens
            usage["out"] += response.usage.output_tokens
        raw = response.content[0].text
    except Exception as e:
        print(f"  [WARN] Claude API error: {e}")
        return {n: n for n in names}

    groups = _extract_json(raw)

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


def run(input_path: str | Path, output_dir: str | Path, model: str,
        keep_raw: bool = False) -> tuple[Path, Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[ERROR] ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)
    client = _anthropic.Anthropic(api_key=api_key)

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
    usage: dict = {"in": 0, "out": 0}

    batches = [all_names[i: i + BATCH_SIZE] for i in range(0, len(all_names), BATCH_SIZE)]
    for idx, batch in enumerate(batches, 1):
        print(f"Normalizing batch {idx}/{len(batches)} ({len(batch)} names) ...", flush=True)
        batch_map = normalize_batch(batch, client, model, context=name_context, freq=name_freq, usage=usage)
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

    nodes_path = output_dir / "nodes_claude.csv"
    edges_path = output_dir / "edges_claude.csv"

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

    cost = usage["in"] * _COST_IN + usage["out"] * _COST_OUT
    print(f"\nNodes: {len(node_ids)}, Edges: {len(edges_final)}")
    print(f"Tokens — in: {usage['in']:,}  out: {usage['out']:,}  cost: ${cost:.4f}")
    print(f"Saved: {nodes_path}")
    print(f"Saved: {edges_path}")
    return nodes_path, edges_path


def main():
    parser = argparse.ArgumentParser(description="KGB name normalizer — Claude API version")
    parser.add_argument("--input", default="data/processed/edges_raw_claude.csv")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--keep-raw", action="store_true",
                        help="Keep edges_raw.csv after normalization")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] {input_path} not found. Run extract_entities.py first.")
        sys.exit(1)

    run(input_path, args.output_dir, args.model, keep_raw=args.keep_raw)


if __name__ == "__main__":
    main()
