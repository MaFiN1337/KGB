"""Compare edges_raw.csv against edges_expected.csv and report Recall/Precision/F1."""
import csv
import sys
import io
import argparse
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent


def load(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def norm(s):
    return s.strip().lower()


def stem(name):
    """First 6 chars of first token — handles Slavic genitive endings."""
    return norm(name).split()[0][:6]


def edge_key(row):
    return (stem(row["source"]), stem(row["target"]), norm(row["sentiment"]), norm(row.get("id", "")))


def main():
    parser = argparse.ArgumentParser(description="Eval edge extraction quality")
    parser.add_argument("--expected", default=str(ROOT / "tests/data/edges_expected.csv"))
    parser.add_argument("--actual", default=str(ROOT / "data/processed/edges_raw.csv"))
    args = parser.parse_args()

    expected = load(args.expected)
    actual   = load(args.actual)

    exp_loose = {edge_key(r): r for r in expected}
    act_loose = {edge_key(r): r for r in actual}

    print(f"Expected: {len(expected)}  Actual: {len(actual)}\n")

    print("=== MATCHED (loose) ===")
    matched_ids = set()
    for k, row in act_loose.items():
        if k in exp_loose:
            matched_ids.add(k)
            quote_ok = (
                stem(row["source"]) in norm(row["evidence_quote"]) or
                stem(row["target"]) in norm(row["evidence_quote"])
            )
            q_flag = "✓" if quote_ok else "✗ quote missing name"
            print(f"  id={row['id']} {row['source'][:18]}→{row['target'][:18]} [{row['sentiment']}] {q_flag}")
            if not quote_ok:
                print(f"    quote: {row['evidence_quote']!r}")

    print("\n=== MISSING (expected but not in actual) ===")
    for k, row in exp_loose.items():
        if k not in act_loose:
            print(f"  id={row['id']} {row['source'][:20]}→{row['target'][:20]} [{row['sentiment']}]")

    print("\n=== EXTRA (actual but not in expected) ===")
    for k, row in act_loose.items():
        if k not in exp_loose:
            print(f"  id={row['id']} {row['source'][:20]}→{row['target'][:20]} [{row['sentiment']}]  quote: {row['evidence_quote'][:60]!r}")

    matched   = len(matched_ids)
    recall    = matched / len(expected) * 100 if expected else 0
    precision = matched / len(actual)   * 100 if actual   else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0

    bad_quotes = sum(
        1 for r in actual
        if not r["evidence_quote"].strip() or (
            stem(r["source"]) not in norm(r["evidence_quote"]) and
            stem(r["target"]) not in norm(r["evidence_quote"])
        )
    )
    print(f"\nRecall={recall:.0f}%  Precision={precision:.0f}%  F1={f1:.0f}%")
    print(f"Bad quotes (empty/no name): {bad_quotes}/{len(actual)}")


if __name__ == "__main__":
    main()
