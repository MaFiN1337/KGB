import io
import json
import csv
import argparse
import re
import os
import sys
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

try:
    import ollama as _ollama
except ImportError:
    print("Run: pip install ollama")
    sys.exit(1)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DEFAULT_MODEL = "llama3.1:8b"

NAME_PROMPT = (
    "List all PERSON names mentioned in this KGB/OGPU archival document (1920s–1930s).\n"
    "Return each name in nominative case (dictionary form).\n"
    "Include everyone: speakers, interrogated persons, people merely mentioned.\n"
    "Include unfamiliar names (Crimean Tatar, Polish, etc.).\n"
    "Exclude organizations, institutions, and job titles without a surname.\n\n"
    "Return ONLY JSON, no explanation:\n"
    "{\"names\": [\"Surname1\", \"Surname2\"]}"
)

SYSTEM_PROMPT = (
    "Extract all relations between people from this KGB/OGPU archival document (1920s–1930s).\n\n"
    "For each relation return:\n"
    "  source       — who is speaking, nominative case\n"
    "  target       — who is being spoken about, nominative case\n"
    "  sentiment    — Звинувачення | Захист | Нейтрально\n"
    "  evidence_quote — first sentence from the document containing target or source name "
    "(or an unambiguous pronoun), copied verbatim, ≤120 chars. "
    "If no such sentence exists, skip this relation entirely.\n\n"
    "Rules:\n"
    "  1. Only real people. Never extract organizations (ОГПУ, КГБ, ВКП(б), Ширкет, etc.).\n"
    "  2. source is the SPEAKER. Use [Автор] when the speaker says «я»/«мене»/«мені».\n"
    "     If the document ends with «Підпис: X» or a rank+name, use that as source.\n"
    "     Example: «Я считаю Котляра неблагонадёжным» → source=[Автор], target=Котляр, Звинувачення.\n"
    "  3. In ВОПРОС/ОТВЕТ format: evidence_quote only from sentences after «ОТВЕТ:», never after «ВОПРОС:».\n"
    "  4. Sentiment — choose one of these three exact strings:\n"
    "     \"Звинувачення\" (ACCUSATION) — use when speaker reports that target:\n"
    "       • committed anti-Soviet acts or made anti-Soviet statements\n"
    "       • was a member of nationalist organizations (Міллі Фірка, etc.)\n"
    "       • had contacts with enemies of Soviet power or political emigrants\n"
    "       • was involved in any politically suspicious activity\n"
    "       This applies even when phrased as a neutral fact: "
    "«Х був членом Міллі Фірки» = Звинувачення.\n"
    "     \"Захист\" (DEFENSE) — speaker explicitly defends target's Soviet loyalty.\n"
    "     \"Нейтрально\" (NEUTRAL) — purely factual: met someone, worked together, location.\n\n"
    "Return ONLY JSON, no explanation:\n"
    "{\"relations\": [{\"source\": \"Name\", \"target\": \"Name\", "
    "\"sentiment\": \"Звинувачення\", \"evidence_quote\": \"...\"}]}"
)

_HEADER_RE = re.compile(
    r"(?:ПОКАЗАНИЯ|ПРОТОКОЛ ДОПРОСА|ПРОДОЛЖЕНИЕ ДОПРОСА|ДОПОЛНИТЕЛЬНЫЕ ПОКАЗАНИЯ)"
    r"\s+([А-ЯЁІЇЄ][А-ЯЁІЇЄ\-]+(?:\s+[А-ЯЁІЇЄ][А-ЯЁІЇЄ]+)?)",
    re.IGNORECASE,
)
_HEADER_ROLES = {
    "обвиняемого", "обвиняемой", "свидетеля", "допрашиваемого",
    "подозреваемого", "гражданина", "гражданки", "по", "существу",
}
_RANK_RE = re.compile(
    r"^(тов\.|тт\.|гр\.|гр-н\b|гр-ка\b|гражданин\b|"
    r"Майор\b|Капитан\b|Лейтенант\b|Полковник\b|Генерал\b|"
    r"Уполномоченный\b|Уполн\.\b|Нач\.\b|начальник\b|"
    r"ПП\b|ВО\b|ОГПУ\b|ГПУ\b|т\.)\s*",
    re.IGNORECASE,
)
_INITIALS_RE = re.compile(r"\s+[А-ЯЇІЄа-яїіє]\.[А-ЯЇІЄа-яїіє]\.?$")
_SINGLE_INITIAL_RE = re.compile(r"\s+[А-ЯЇІЄа-яїіє]\.?$")
_PATRONYMIC_RE = re.compile(
    r"^[А-ЯЁІЇЄ][а-яёіїє]+(ович|евич|євич|овна|евна|євна|івна|йович)$",
    re.IGNORECASE,
)

_SKIP_NAMES = {
    "я", "он", "она", "они", "мы", "вы", "ты",
    "его", "её", "их", "себя", "ему", "ей",
    "свидетель", "обвиняемый", "допрашиваемый", "следователь",
    "уполномоченный",
    "[автор]",  # unresolved speaker placeholder
}

# Organizations and non-person tokens that LLMs repeatedly hallucinate as names
_ORG_BLOCKLIST: set[str] = {
    # secret police
    "огпу", "кгб", "нквд", "гпу",
    # party/state bodies
    "вкп", "вкп(б)", "ркп", "цк", "снк", "цик", "ц.и.к", "ц.и.к.", "татбюро",
    "наркозем", "наркомат", "наркомісте", "наркмоисте",
    # Crimean institutions
    "міллі фірка", "милли фирка", "милли-фирка",
    "ширкет", "ширкета",
    # misc abbreviations that are not names
    "ква", "по",
}


class Relation(BaseModel):
    source: str
    target: str
    sentiment: Literal["Захист", "Звинувачення", "Нейтрально"]
    evidence_quote: str


class DocumentResponse(BaseModel):
    relations: list[Relation]


def detect_speaker_regex(text: str) -> str | None:
    m = _HEADER_RE.search(text)
    if not m:
        return None
    words = m.group(1).strip().split()
    while words and words[0].lower() in _HEADER_ROLES:
        words.pop(0)
    return " ".join(words[:2]) if words else None


def _clean_name(raw: str) -> str:
    raw = raw.strip()
    m = re.match(r"\[Автор:\s*([^\]]+)\]", raw, re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
    name = _RANK_RE.sub("", raw)
    name = _INITIALS_RE.sub("", name)
    name = _SINGLE_INITIAL_RE.sub("", name)
    return name.strip()


def is_valid_name(name: str) -> bool:
    cleaned = _clean_name(name).strip()
    if len(cleaned) < 3:
        return False
    low = cleaned.lower()
    if low in _SKIP_NAMES:
        return False
    if low in _ORG_BLOCKLIST:
        return False
    if not re.search(r"[А-ЯЁа-яёІіЇїЄє]", cleaned):
        return False
    if _PATRONYMIC_RE.match(cleaned.split()[0]):
        return False
    return True


def _extract_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"relations": []}


def _find_evidence(text: str, name: str) -> str:
    pattern = re.compile(re.escape(name.split()[0]), re.IGNORECASE)
    for chunk in re.findall(r"ОТВЕТ[:\s]+(.*?)(?=ВОПРОС[:\s]|$)", text, re.DOTALL | re.IGNORECASE):
        for sent in re.split(r"(?<=[.!?])\s+", chunk):
            if pattern.search(sent):
                return sent.strip()[:120]
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if pattern.search(sent):
            return sent.strip()[:120]
    return ""


def _fill_missing(
    relations: list[Relation],
    known_names: list[str],
    text: str,
    speaker: str | None,
) -> list[Relation]:
    if not speaker:
        return relations
    mentioned = {r.source.lower() for r in relations} | {r.target.lower() for r in relations}
    extra = []
    for name in known_names:
        if name.lower() in mentioned or name.lower() == speaker.lower():
            continue
        evidence = _find_evidence(text, name)
        if not evidence:
            continue
        extra.append(Relation(source=speaker, target=name, sentiment="Нейтрально", evidence_quote=evidence))
    return relations + extra


def extract_names(text: str, model: str, threads: int) -> list[str]:
    try:
        resp = _ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": NAME_PROMPT},
                {"role": "user", "content": text[:3500]},
            ],
            format="json",
            options={"temperature": 0.0, "num_ctx": 4096, "num_predict": 256, "num_thread": threads},
        )
        data = _extract_json(resp.message.content)
        raw = data.get("names", [])
        return [_clean_name(n) for n in raw if isinstance(n, str) and is_valid_name(n)]
    except Exception:
        return []


def call_llm(
    text: str,
    current_speaker: str | None,
    model: str,
    threads: int,
    known_names: list[str] | None = None,
) -> DocumentResponse:
    speaker_line = f"[Автор: {current_speaker}]\n\n" if current_speaker else ""
    names_line = ("Відомі особи у тексті (лише для довідки, НЕ джерело цитат): " + ", ".join(known_names) + "\n\n") if known_names else ""
    try:
        resp = _ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": speaker_line + names_line + text[:3500]},
            ],
            format="json",
            options={"temperature": 0.1, "num_ctx": 4096, "num_predict": 512, "num_thread": threads},
        )
        data = _extract_json(resp.message.content)
        return DocumentResponse(**data)
    except Exception:
        return DocumentResponse(relations=[])


def main():
    parser = argparse.ArgumentParser(description="KGB archive NLP entity extractor — pass 1")
    parser.add_argument("--input", default="data/interim/documents.json")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N documents")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N documents")
    parser.add_argument("--threads", type=int, default=6,
                        help="CPU threads for Ollama inference (default: 6)")
    parser.add_argument("--keep-raw", action="store_true",
                        help="Keep edges_raw.csv after normalization")
    args = parser.parse_args()

    try:
        _ollama.list()
    except Exception as e:
        print(f"[ERROR] Ollama not reachable: {e}")
        print("Make sure Ollama is running: ollama serve")
        print(f"And the model is pulled: ollama pull {args.model}")
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        documents = json.load(f)

    documents = documents[args.offset:]
    if args.limit:
        documents = documents[:args.limit]

    os.makedirs(args.output_dir, exist_ok=True)

    all_edges: list[dict] = []
    skipped = 0
    current_speaker: str | None = None

    for i, doc in enumerate(documents, 1):
        doc_id = str(doc["id"])
        text = doc.get("full_text", "").strip()

        if len(text) < 150:
            print(f"[{i}/{len(documents)}] SKIP  id={doc_id}")
            continue

        header_speaker = detect_speaker_regex(text)
        if header_speaker:
            current_speaker = header_speaker

        print(f"[{i}/{len(documents)}] id={doc_id}", end=" ... ", flush=True)

        known_names = extract_names(text, args.model, args.threads)
        print(f"names={known_names}", end=" | ", flush=True)

        result = call_llm(text, current_speaker, args.model, args.threads, known_names)

        relations = _fill_missing(result.relations, known_names, text, current_speaker)

        print(f"speaker={current_speaker or '?'}", end=" | ", flush=True)

        count = 0
        for rel in relations:
            src = _clean_name(rel.source)
            tgt = _clean_name(rel.target)
            if not is_valid_name(src) or not is_valid_name(tgt):
                skipped += 1
                continue
            if src.lower() == tgt.lower():
                continue
            all_edges.append({
                "source": src,
                "target": tgt,
                "sentiment": rel.sentiment,
                "evidence_quote": rel.evidence_quote,
                "id": doc_id,
            })
            count += 1

        print(f"{count} relations")

    edges_raw_path = Path(args.output_dir) / "edges_raw_ollama.csv"
    with open(edges_raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "target", "sentiment", "evidence_quote", "id"])
        w.writeheader()
        w.writerows(all_edges)

    print(f"\nEdges (raw): {len(all_edges)}")
    print(f"Skipped (pronouns/roles): {skipped}")
    print(f"\nRunning name normalization ...", flush=True)

    normalize_cmd = [
        sys.executable,
        str(Path(__file__).parent / "normalize_names.py"),
        "--input", str(edges_raw_path),
        "--output_dir", args.output_dir,
        "--model", args.model,
    ]
    if args.keep_raw:
        normalize_cmd.append("--keep-raw")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    subprocess.run(normalize_cmd, check=True, env=env)


if __name__ == "__main__":
    main()
