# KGB Repression Semantic Graph

Automatically transforms raw XML annotations of KGB archival cases into an interactive graph that visualizes who testified against whom **(red links)** and who defended whom **(green links)**.

---

## Overview

This project processes scanned KGB archive documents from the State Archive of the Security Service of Ukraine (ГДАСБУ). Using OCR annotations, NLP entity extraction, and graph visualization, we reconstruct the hidden social network of loyalty and accusations inside Soviet-era repression cases.

**4 archival cases processed:**
- `ф5_оп1_спр8441` — 107 pages
- `ф6_оп1_спр51115фп` — 90 pages
- `ф13_оп1_спр1121` — 108 pages
- `ф16_оп1_спр705` — 18 pages

---

## Pipeline

```
annotations.xml  →  [parse_xml]  →  documents.json
                                          ↓
                               [extract_entities]  ←── Ollama (llama3.1:8b)
                                          ↓
                               [normalize_names]   ←── Ollama (llama3.1:8b)
                                          ↓
                              nodes.csv + edges.csv
                                          ↓
                                  [graph builder]
                                          ↓
                               graph.html (interactive)
```

`extract_entities` and `normalize_names` run as a single command — normalization is triggered automatically at the end of extraction. Use `--keep-raw` to retain the intermediate `edges_raw.csv`.

---

## Project Structure

```
KGB/
├── data/
│   ├── raw/                        # original XML — do not modify
│   │   └── full_annotation.xml
│   └── interim/                    # output from parse_xml
│       └── documents.json
├── src/
│   ├── merge_CVAT.py               # merge multi-annotator CVAT exports
│   ├── parse_xml.py                # XML → documents.json
│   ├── extract_entities.py         # pass 1: LLM relation extractor
│   ├── normalize_names.py          # pass 2: LLM name normalizer
│   └── build_graph.py               # graph generator
├── output/
│   └── graph.html                  # interactive graphs
├── requirements.txt
└── README.md
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
# Requires Ollama running locally: 
irm https://ollama.com/install.ps1 | iex
ollama pull llama3.1:8b
```

**2. Parse XML annotations**
```bash
python src/parse_xml.py --input data/raw/full_annotation.xml --output data/interim/documents.json
```

**3. Extract entities + normalize names**
```bash
# Full dataset — produces data/processed/nodes.csv and edges.csv
python src/extract_entities.py --input data/interim/documents.json

# Test dataset
python src/extract_entities.py --input tests/data/documents_test.json --output_dir data/processed

# Keep intermediate edges_raw.csv for debugging
python src/extract_entities.py --input data/interim/documents.json --keep-raw
```

---

## Graph Logic

| Element | Rule |
|---|---|
| 🟢 Green edge | Sentiment == "Захист" (Defense) |
| 🔴 Red edge | Sentiment == "Звинувачення" (Accusation) |
| ⚫ Grey edge | Sentiment == "Нейтрально" (Neutral) |
| Node size | Degree centrality — more connections = bigger node |
| Tooltip | Hover over edge → shows original quote from document |

---

## Team

| Name | Role | Responsibility |
|---|---|---|
| Valentyna Dermenzhy | **A — Data Engineer** | XML parsing, text reconstruction, JSON handover |
| Oleksandra Malii | **B — NLP Engineer** | Entity extraction via LLM, name normalization |
| Anton Pihuliak | **C — Graph Scientist** | Network construction, interactive visualization |
