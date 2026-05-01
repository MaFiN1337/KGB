import io
import json
import csv
import argparse
import re
import os
import sys
import subprocess
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from typing import Literal

from pydantic import BaseModel

try:
    import ollama as _ollama
except ImportError:
    print("Run: pip install ollama")
    sys.exit(1)

DEFAULT_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = (
    "Ти аналізуєш документи архіву КГБ/ОГПУ 1920–1930-х років.\n\n"
    "Знайди всі зв'язки між конкретними людьми. Для кожного зв'язку поверни:\n"
    "  source — хто говорить (називний відмінок)\n"
    "  target — про кого йдеться (називний відмінок)\n"
    "  sentiment — «Захист» | «Звинувачення» | «Нейтрально»\n"
    "  evidence_quote — серед речень тексту знайди ПЕРШЕ, де є прізвище target або source,\n"
    "    і скопіюй його дослівно (≤120 символів).\n"
    "    Займенник («його», «її», «мене», «ним») НЕ є прізвищем.\n"
    "    Якщо такого речення немає — запис не створюй.\n\n"
    "Правила:\n"
    "  1. Тільки реальні люди (прізвище або ім'я). "
    "НЕ витягуй: ОГПУ, КГБ, партія, ВКП(б), влада, держава, завод, цех тощо.\n"
    "  2. source — людина, що ГОВОРИТЬ.\n"
    "     «я»/«мене»/«мені» або цитата, що починається з «Я» → source=[Автор]; без [Автор] — пропусти.\n"
    "     Якщо в кінці документа є «Підпис: X» або «Майор X» — source=X (ігноруй [Автор]).\n"
    "     Приклади:\n"
    "       «Я считаю Котляра неблагонадёжным» → source=[Автор], target=Котляр, Звинувачення.\n"
    "       «Філонюк обмовляє мене»            → source=[Автор], target=Філонюк, Звинувачення.\n"
    "  3. «він»/«вона» — лише якщо однозначно хто; «ми»/«ти»/«ви» — пропускай.\n"
    "  4. Одне речення може стосуватися двох людей — тоді два записи:\n"
    "     «Сітдіков наклепував на Котляра» →\n"
    "       [Автор]→Сітдіков Звинувачення (злочинець),\n"
    "       [Автор]→Котляр Захист (жертва наклепу).\n"
    "     Але якщо другий — «мене» (тобто [Автор]) — другий запис НЕ потрібен.\n"
    "  5. ВОПРОС/ОТВЕТ: evidence_quote — лише речення після «ОТВЕТ:». Ніколи після «ВОПРОС:».\n"
    "  6. НЕ використовуй рядки «Підпис:» / «Подпись:» / «Підписав:» як evidence_quote.\n"
    "     Якщо документ адресовано («Начальнику X» / «тов. X»): target=X, Нейтрально,\n"
    "     evidence_quote = рядок адреси («Начальнику ... тов. X»).\n"
    "  7. «Захист» — автор ЯВНО захищає або виправдовує особу ('чесний громадянин', 'безпідставно' тощо).\n"
    "     Якщо автор цитує або повідомляє про антирадянські слова/дії особи — це ЗАВЖДИ «Звинувачення»,\n"
    "     навіть якщо автор від цих дій дистанціюється («Я не підтримував» тощо).\n"
    "  8. «Нейтрально» — будь-яка згадка прізвища без оцінки. "
    "Витягуй ВСІХ людей з тексту, включаючи тих, хто просто присутній чи згаданий.\n"
    "     Якщо в тексті про особу сказано «прізвища не знаю» або «якийсь» — пропускай.\n\n"
    "Повернути ТІЛЬКИ JSON без пояснень:\n"
    "{\"relations\": ["
    "{\"source\": \"Прізвище\", \"target\": \"Прізвище\", "
    "\"sentiment\": \"Звинувачення\", "
    "\"evidence_quote\": \"речення з прізвищем\"}]}"
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

_SKIP_NAMES = {
    "я", "он", "она", "они", "мы", "вы", "ты",
    "его", "её", "их", "себя", "ему", "ей",
    "свидетель", "обвиняемый", "допрашиваемый", "следователь",
    "уполномоченный",
}

# standalone patronymics like "Васильович", "Олексіївна" — not valid person names
_PATRONYMIC_RE = re.compile(
    r"^[А-ЯЁІЇЄ][а-яёіїє]+(ович|евич|євич|овна|евна|євна|івна|йович|йович)$",
    re.IGNORECASE,
)


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


def _doc_sort_key(doc: dict) -> int:
    return int(doc.get("id", 0))


def _clean_name(raw: str) -> str:
    name = _RANK_RE.sub("", raw.strip())
    name = _INITIALS_RE.sub("", name)
    name = _SINGLE_INITIAL_RE.sub("", name)
    return name.strip()


def is_valid_name(name: str) -> bool:
    cleaned = _clean_name(name).strip()
    if len(cleaned) < 3:
        return False
    if cleaned.lower() in _SKIP_NAMES:
        return False
    if not re.search(r"[А-ЯЁа-яёІіЇїЄє]", cleaned):
        return False
    # reject bare patronymics without a preceding surname
    first_word = cleaned.split()[0]
    if _PATRONYMIC_RE.match(first_word):
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


def call_llm(text: str, current_speaker: str | None, model: str) -> DocumentResponse:
    speaker_line = f"[Автор: {current_speaker}]\n\n" if current_speaker else ""
    try:
        resp = _ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": speaker_line + text[:3500]},
            ],
            format="json",
            options={"temperature": 0.1},
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--keep-raw", action="store_true",
                        help="Keep edges_raw.csv intermediate file after normalization")
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

    documents.sort(key=_doc_sort_key)
    documents = documents[args.offset:]
    if args.limit:
        documents = documents[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)

    all_edges: list[dict] = []
    skipped = 0
    current_speaker: str | None = None

    for i, doc in enumerate(documents, 1):
        doc_id = str(doc.get("id", doc.get("doc_id", i)))
        text = doc.get("full_text", "").strip()

        if len(text) < 50:
            print(f"[{i}/{len(documents)}] SKIP  id={doc_id}")
            continue

        header_speaker = detect_speaker_regex(text)
        if header_speaker:
            current_speaker = header_speaker

        print(f"[{i}/{len(documents)}] id={doc_id}", end=" ... ", flush=True)

        result = call_llm(text, current_speaker, args.model)

        print(f"speaker={current_speaker or '?'}", end=" | ", flush=True)

        count = 0
        for rel in result.relations:
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

    edges_raw_path = Path(args.output_dir) / "edges_raw.csv"
    with open(edges_raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["source", "target", "sentiment", "evidence_quote", "id"],
        )
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
