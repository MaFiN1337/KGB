import xml.etree.ElementTree as ET
import pandas as pd
import json
import argparse
import re
import os



TEXT_LABELS = {
    "printed_text":     "Printed Text",
    "handwritten_text": "Contented Text",
    "date_handwritten": "Date Written Text",
    "date_printed":     "Date Printed Text",
}

SKIP_LABELS = {"seal", "signature", "table"}
MIN_TEXT_LENGTH = 3


def is_garbage(text: str) -> bool:
    """Returns true if text is garbage (page number, empty, etc.)"""
    text = text.strip()

    if not text:
        return True

    if len(text) < MIN_TEXT_LENGTH:
        return True

    if re.fullmatch(r'\d{1,4}', text):
        return True

    return False



def parse_annotations(xml_path: str, limit: int = None) -> list[dict]:
    """
    Parses XML file and returns a list of documents.
    Each document: {"doc_id": "...", "full_text": "..."}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results = []

    for image in root.findall("image"):
        image_name = image.get("name")
        rows = []

        for box in image.findall("box"):
            label = box.get("label")

            if label in SKIP_LABELS:
                continue

            if label not in TEXT_LABELS:
                continue

            ytl = float(box.get("ytl", 0))
            xtl = float(box.get("xtl", 0))

            attr_name = TEXT_LABELS[label]
            text = ""
            for attr in box.findall("attribute"):
                if attr.get("name") == attr_name:
                    text = (attr.text or "").strip()
                    break

            if is_garbage(text):
                continue

            rows.append({
                "ytl": ytl,
                "xtl": xtl,
                "text": text,
                "label": label,
            })

        if not rows:
            print(f"  [WARNING] Image {image_name} — no text found")
            continue

        df = pd.DataFrame(rows)
        df = df.sort_values(by=["ytl", "xtl"]).reset_index(drop=True)

        full_text = " ".join(df["text"].tolist())

        results.append({
            "id": len(results),
            "doc_id": image_name,
            "full_text": full_text,
        })

        print(f"  [OK] {image_name} — {len(df)} text blocks")

        if limit and len(results) >= limit:
            break

    return results


def main():
    parser = argparse.ArgumentParser(description="KGB archive XML annotation parser")
    parser.add_argument(
        "--input",
        default="annotations.xml",
        help="Path to XML file (default: annotations.xml)"
    )
    parser.add_argument(
        "--output",
        default="documents.json",
        help="Path to output JSON file (default: documents.json)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents (e.g. --limit 10)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        return

    print(f"\nReading file: {args.input}")
    print("-" * 50)

    documents = parse_annotations(args.input, limit=args.limit)

    print("-" * 50)
    print(f"Documents processed: {len(documents)}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {args.output}")

    if documents:
        print("\nPreview of first document:")
        print(f"  doc_id: {documents[0]['doc_id']}")
        print(f"  full_text: {documents[0]['full_text'][:300]}...")


if __name__ == "__main__":
    main()