"""RAG по официальным документам для приезжих (docs/*.md).

Без внешних сервисов и API-ключей: поиск TF-IDF (слова + символьные n-граммы),
ответ собирается из самых близких предложений документов, всегда со ссылкой на источник.
Если близость ниже порога, отвечает «не знаю». Попытки обойти правила отклоняются.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DEFAULT_DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# Подобраны на тестовых вопросах docs/rag_test_questions.csv (см. tests/test_rag.py)
MIN_SCORE = 0.16        # ниже этой близости отвечаем «не знаю»
CONFIDENT_SCORE = 0.23  # от этой близости ответ считается прямым; между порогами «связанный фрагмент»
TOP_CHUNKS = 3
MAX_SENTENCES = 3
MAX_QUESTION_LEN = 500

UNKNOWN_TEXT = ("Не знаю: в моих документах для приезжих нет ответа на этот вопрос. "
                "Уточните на официальных сайтах.")
RELATED_TEXT = ("Прямого ответа на ваш вопрос в документах не нашёл. Ближайший по теме фрагмент "
                "(проверьте, отвечает ли он на ваш вопрос):")
REFUSAL_TEXT = ("Я отвечаю только по официальным документам для приезжих и не меняю свои правила "
                "и не раскрываю служебные инструкции. Задайте вопрос о визе, транспорте, жилье, "
                "безопасности или налогах в Турции.")

# Попытки обойти правила (русский и английский)
INJECTION_PATTERNS = [
    r"забудь\w*\s+(все\s+|свои\s+|предыдущ\w+\s+)?(правил|инструкц)",
    r"игнорируй\w*\s+(все\s+|свои\s+|предыдущ\w+\s+)?(правил|инструкц|документ|путеводител)",
    r"забудь\w*\s+правило",
    r"придума\w+",
    r"(системн\w+|скрыт\w+|служебн\w+)\s+(инструкц|промпт|правил)",
    r"\bignore\b.{0,40}\b(rules|instructions|guide|documents|previous)\b",
    r"\bforget\b.{0,40}\b(rules|instructions|guide|documents)\b",
    r"(system|hidden|secret)\s+(prompt|instructions?)",
    r"(print|reveal|show)\b.{0,40}\b(instructions|prompt)\b",
    r"\bjailbreak\b|\bdeveloper mode\b",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)

# Английские слова, которых нет в русских документах: расширяем запрос русскими основами
EN_RU = {
    "station": "станция", "stations": "станции", "change": "пересадка", "transfer": "пересадка",
    "passport": "паспорт", "lost": "потеря", "emergency": "экстренная", "document": "документ",
    "temporary": "временный", "travel": "поездка", "return": "возвращение", "home": "домой",
    "citizen": "гражданин", "metro": "метро", "tax": "налог", "visa": "виза", "airport": "аэропорт",
    "ticket": "билет", "fare": "стоимость", "price": "цена", "cost": "стоит", "rent": "аренда",
    "permit": "разрешение", "hotel": "отель", "police": "полиция", "ambulance": "скорая",
    "free": "бесплатно", "refund": "возврат", "card": "карта", "bus": "автобус",
    "kyrgyz": "кыргызстан", "kyrgyzstan": "кыргызстан", "european": "европейская",
    "side": "сторона", "toward": "в сторону", "which": "какой", "where": "где", "how": "как",
    "much": "сколько", "when": "когда", "accommodation": "размещение", "rate": "ставка",
    "theft": "кража", "pickpocket": "кража", "pickpockets": "кража", "fire": "пожар",
    "host": "хозяин", "apartment": "квартира", "flat": "квартира", "guest": "гость",
    "guests": "гости", "money": "деньги", "bank": "банк", "safe": "безопасность",
    "safety": "безопасность", "customs": "таможня", "vat": "НДС", "stay": "пребывание",
    "days": "дней", "register": "регистрация", "registration": "регистрация",
}


# Русские слова из вопросов, которых нет в документах в такой форме: добавляем близкие слова
RU_SYN = {
    "стоит": "стоимость цена тариф", "стоимость": "стоит цена", "проезд": "билет проезд",
    "добраться": "маршрут доехать проехать", "доехать": "маршрут добраться", "звонить": "номер телефон",
    "скорая": "медицинская помощь 112", "скорую": "медицинская помощь 112",
    "ambulance": "скорая медицинская помощь 112 экстренная", "call": "звонить номер",
    "tourist": "туристический налог", "tax": "налог", "airport": "аэропорт",
}


@dataclass
class Chunk:
    text: str
    file: str
    title: str
    source_name: str
    source_url: str
    date_checked: str


@dataclass
class Answer:
    status: str                     # "answered" | "related" | "unknown" | "refused"
    text: str
    score: float = 0.0
    sources: list[dict] = field(default_factory=list)


def docs_dir() -> Path:
    return Path(os.environ.get("ISTANBUL_DOCS_DIR", DEFAULT_DOCS_DIR))


def _parse(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n(.*)", raw, re.S)
    if not m:
        raise ValueError(f"{path.name}: нет шапки с источником (--- ... ---).")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    for key in ("title", "source_name", "source_url", "date_checked"):
        if not meta.get(key):
            raise ValueError(f"{path.name}: в шапке нет поля {key}.")
    return meta, m.group(2).strip()


def load_chunks(directory: Path | None = None) -> list[Chunk]:
    """Читает docs/NN_*.md и режет на абзацы. Бросает понятную ошибку, если документов нет."""
    d = Path(directory) if directory else docs_dir()
    files = sorted(d.glob("[0-9][0-9]_*.md"))
    if not files:
        raise FileNotFoundError(f"В {d} нет документов NN_*.md для RAG.")
    chunks = []
    for f in files:
        meta, body = _parse(f)
        for para in re.split(r"\n\s*\n", body):
            para = re.sub(r"\s+", " ", para).strip()
            if len(para) >= 40:
                chunks.append(Chunk(para, f.name, meta["title"], meta["source_name"],
                                    meta["source_url"], meta["date_checked"]))
    return chunks


def _expand_en(question: str) -> str:
    """Добавляет русские слова к английским словам запроса."""
    words = re.findall(r"\w+", question.lower())
    extra = [EN_RU[w] for w in words if w in EN_RU]
    return question + " " + " ".join(extra) if extra else question


def _expand(question: str) -> str:
    """Расширение запроса для поиска: перевод слов-подсказок плюс близкие русские слова."""
    base = _expand_en(question)
    words = re.findall(r"\w+", question.lower())
    extra = [RU_SYN[w] for w in words if w in RU_SYN]
    return base + " " + " ".join(extra) if extra else base


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[А-ЯA-ZЁ0-9«\"])", text)
    return [p.strip() for p in parts if len(p.strip()) > 15]


STOPWORDS = set("и в во не на с со по к о об от для что как это или а но же ли бы то из у за при до "
                "под над про без есть какой какая какие каком каких можно нужно the a an of to in "
                "is are do does can i my".split())
STEM_LEN = 5  # грубая «основа» слова: обрезка до 5 символов, чтобы склонения совпадали


# Слова-«пустышки» вопроса: по ним нельзя судить, что абзац отвечает на вопрос
GENERIC = set("сколько стоит стоимость цена цены какой какая какие как где когда можно нужно нужна "
              "ли что такое есть быть будет делать надо how what which where when much many does "
              "there tell please стамбул стамбуле стамбула турции турция".split())


def _stem_text(text: str) -> str:
    words = re.findall(r"\w+", text.lower())
    return " ".join(w[:STEM_LEN] for w in words if w not in STOPWORDS and len(w) > 1)


class RagIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self._encode([f"{c.title}. {c.text}" for c in chunks], fit=True)

    def _encode(self, texts, fit=False):
        stemmed = [_stem_text(t) for t in texts]
        m = self.vec.fit_transform(stemmed) if fit else self.vec.transform(stemmed)
        return normalize(m.tocsr())

    def search(self, question: str, k: int = TOP_CHUNKS) -> list[tuple[float, Chunk]]:
        q = self._encode([_expand(question)])
        scores = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in order]


def load_index(directory: Path | None = None) -> RagIndex:
    return RagIndex(load_chunks(directory))


def is_injection(question: str) -> bool:
    return bool(_INJECTION.search(question))


def _key_stems(question: str) -> set[str]:
    """Основы содержательных слов самого вопроса (без слов-пустышек и добавленных синонимов)."""
    words = re.findall(r"\w+", _expand_en(question).lower())
    for w in list(words):
        if w not in GENERIC and w in RU_SYN:  # синонимы пустышек («стоит») не считаем
            words += RU_SYN[w].split()
    return {w[:STEM_LEN] for w in words if len(w) > 2 and w not in STOPWORDS and w not in GENERIC}


def _chunk_has_key(question: str, chunk: Chunk) -> bool:
    keys = _key_stems(question)
    if not keys:
        return True
    have = set(_stem_text(f"{chunk.title}. {chunk.text}").split())
    return bool(keys & have)


def _pick_sentences(index: RagIndex, question: str, chunk: Chunk) -> list[str]:
    sents = _sentences(chunk.text)
    if len(sents) <= MAX_SENTENCES:
        return sents
    q = index._encode([_expand(question)])
    s = (index._encode(sents) @ q.T).toarray().ravel()
    keep = sorted(np.argsort(-s)[:MAX_SENTENCES])
    return [sents[i] for i in keep]


def answer(question: str, index: RagIndex, min_score: float = MIN_SCORE) -> Answer:
    """Отвечает по документам. Статусы: answered / related (слабая связь) / unknown / refused."""
    q = (question or "").strip()
    if len(q) < 3:
        return Answer("unknown", "Вопрос слишком короткий. Сформулируйте его подробнее.")
    if len(q) > MAX_QUESTION_LEN:
        return Answer("unknown", f"Вопрос слишком длинный (больше {MAX_QUESTION_LEN} символов).")
    if is_injection(q):
        return Answer("refused", REFUSAL_TEXT)
    hits = index.search(q)
    best_score, best = hits[0]
    if best_score < min_score or not _chunk_has_key(q, best):
        return Answer("unknown", UNKNOWN_TEXT, best_score)
    parts, sources, seen = [], [], set()
    for score, ch in hits:
        if score < min_score or (parts and score < best_score * 0.6):
            continue
        parts.extend(_pick_sentences(index, q, ch)[:MAX_SENTENCES] if not parts else
                     _pick_sentences(index, q, ch)[:2])
        if ch.file not in seen:
            seen.add(ch.file)
            sources.append({"file": ch.file, "title": ch.title, "source_name": ch.source_name,
                            "url": ch.source_url, "date": ch.date_checked,
                            "score": round(score, 3)})
        if len(sources) >= 2:
            break
    text = " ".join(parts)
    if best_score < CONFIDENT_SCORE:
        return Answer("related", f"{RELATED_TEXT}\n\n{text}", best_score, sources)
    return Answer("answered", text, best_score, sources)


def format_sources(sources: list[dict]) -> str:
    return "\n".join(f"- {s['source_name']}: {s['title']} ({s['url']}), проверено {s['date']}"
                     for s in sources)
