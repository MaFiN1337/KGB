import io
import json
import csv
import argparse
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import ollama as _ollama
except ImportError:
    print("Run: pip install ollama")
    sys.exit(1)

DEFAULT_MODEL = "llama3.1:8b"
BATCH_SIZE = 60

NORMALIZE_PROMPT = (
    "Ти аналізуєш список імен людей з архівних документів КГБ/ОГПУ 1920–1930-х років.\n\n"
    "Одна людина може зустрічатися по-різному:\n"
    "  — різні відмінки: «КОТЛЯРА», «Котляру», «Котляр»\n"
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
    """First 4 Cyrillic chars of first word — used to validate LLM merge proposals."""
    first = name.strip().split()[0] if name.strip() else name
    return re.sub(r"[^а-яёіїєА-ЯЁІЇЄa-z]", "", first).lower()[:4]


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
                    context: dict[str, str] | None = None) -> dict[str, str]:
    lines = []
    for n in names:
        ctx = context.get(n, "") if context else ""
        lines.append(f"- {n}" + (f"  (приклад з тексту: «{ctx}»)" if ctx else ""))
    names_text = "\n".join(lines)
    try:
        resp = _ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": NORMALIZE_PROMPT},
                {"role": "user", "content": names_text},
            ],
            format="json",
            options={"temperature": 0.0},
        )
        groups = _extract_json(resp.message.content)
    except Exception as e:
        print(f"  [WARN] LLM error: {e}")
        return {n: n for n in names}

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
            validated[k] = k
        elif _stem4(k) == _stem4(v):
            if v in names_set:
                # canonical exists in our input — use as-is
                validated[k] = v
            else:
                # LLM invented a full-name expansion not in input;
                # pick best nominative form from same-stem input names
                same_stem = [n for n in names_set if _stem4(n) == _stem4(v)]
                validated[k] = _best_canonical(same_stem) if same_stem else k
        else:
            validated[k] = k  # wrong surname stem — reject entirely
    return validated


def _best_canonical(candidates: list[str]) -> str:
    """Among candidates, prefer nominative form over genitive, then longest."""
    def score(n: str) -> tuple:
        last = n.lower()[-1] if n else ""
        # Ukrainian/Russian genitive surname endings: -а, -я (but not -ко, -нко, -о)
        gen = last in ("а", "я") and not n.lower().endswith(("ко", "нко"))
        return (not gen, len(n))
    return max(candidates, key=score)


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
    for edge in edges_raw:
        for field in ("source", "target"):
            name = edge[field].strip()
            if name and name not in name_context and edge.get("evidence_quote"):
                name_context[name] = edge["evidence_quote"][:80]

    full_mapping: dict[str, str] = {}

    batches = [all_names[i: i + BATCH_SIZE] for i in range(0, len(all_names), BATCH_SIZE)]
    for idx, batch in enumerate(batches, 1):
        print(f"Normalizing batch {idx}/{len(batches)} ({len(batch)} names) ...", flush=True)
        batch_map = normalize_batch(batch, model, context=name_context)
        full_mapping.update(batch_map)
        for raw, canonical in sorted(batch_map.items()):
            if raw != canonical:
                print(f"  {raw!r} → {canonical!r}")

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

    nodes_path = output_dir / "nodes.csv"
    edges_path = output_dir / "edges.csv"

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
    parser.add_argument("--input", default="data/processed/edges_raw.csv")
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
