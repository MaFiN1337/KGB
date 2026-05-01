# KGB Repression Semantic Graph

Automatically transforms raw XML annotations of KGB archival cases into an interactive graph that visualizes who testified against whom **(red links)** and who defended whom **(green links)**.

---

## Overview

This project processes scanned KGB archive documents from the State Archive of the Security Service of Ukraine (ГДАСБУ). Using OCR annotations, NLP entity extraction, and graph visualization, we reconstruct the hidden social network of loyalty and accusations inside Soviet-era repression cases.

**4 archival cases processed:**
- `ф5_оп1_спр8441` — 97 pages
- `ф6_оп1_спр51115фп` — 90 pages
- `ф13_оп1_спр1121` — 108 pages
- `ф16_оп1_спр705` — 18 pages

---

## Pipeline

```
annotations.xml  →  [A: Parser]  →  documents.json
                                         ↓
                               [B: NLP Extractor]
                                         ↓
                          nodes.csv + edges.csv
                                         ↓
                              [C: Graph Builder]
                                         ↓
                               graph.html (interactive)
```

---

## Project Structure

```
KGB/
├── data/
│   ├── raw/                  # original XML — do not modify
│   │   └── full_annotation.xml
│   ├── interim/              # output from Participant A
│   │   └── documents.json
│   └── processed/            # output from Participant B
│       ├── nodes.csv
│       └── edges.csv
├── src/
│   ├── parse_xml.py          # A: XML parser
│   ├── validate.py           # A: output validator
│   ├── extract_entities.py   # B: NLP entity extractor
│   └── build_graph.py        # C: graph builder
├── output/
│   └── graph.html            # final interactive graph
├── notebooks/                # experiments only
├── requirements.txt
└── README.md
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Parse XML annotations (Participant A)**
```bash
# Full dataset
python src/parse_xml.py --input data/raw/full_annotation.xml --output data/interim/documents.json

# Test with 10 documents
python src/parse_xml.py --input data/raw/full_annotation.xml --output data/interim/documents_10.json --limit 10

```

**3. Extract entities (Participant B)**
```bash
python src/extract_entities.py --input data/interim/documents.json
```

**4. Build graph (Participant C)**
```bash
python src/build_graph.py
# opens output/graph.html
```

---

## Graph Logic

| Element | Rule |
|---|---|
| 🟢 Green edge | Sentiment == "Defense" |
| 🔴 Red edge | Sentiment == "Accusation" |
| ⚫ Grey edge | Sentiment == "Neutral" |
| Node size | Degree centrality — more connections = bigger node |
| Tooltip | Hover over edge → shows original quote from document |

---

## Team

| Role | Responsibility |
|---|---|
| **A — Data Engineer** | XML parsing, text reconstruction, JSON handover |
| **B — NLP Engineer** | Entity extraction via LLM, name normalization |
| **C — Graph Scientist** | Network construction, interactive visualization |