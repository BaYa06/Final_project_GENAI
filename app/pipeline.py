"""Логика приложения: маскирование текста, NLP-классификатор, оценка цены.

Признаки здесь должны совпадать с ноутбуком (ячейка 5 и 11). Модели загружаются из
папки models/ (joblib-файлы из архива, который делает ячейка 11 ноутбука).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Границы, вне которых вход считается ошибкой (защита от невозможных значений)
LIMITS = {
    "accommodates": (1, 16),
    "bedrooms": (0, 10),
    "baths": (0, 10),
    "review_score": (0.0, 5.0),
    "n_amenities": (0, 200),
    "latitude": (40.5, 41.9),
    "longitude": (27.5, 30.5),
}
MIN_TEXT_LEN = 20
MIN_KNOWN_TOKENS = 3


def models_dir() -> Path:
    return Path(os.environ.get("ISTANBUL_MODELS_DIR", DEFAULT_MODELS_DIR))


def load_artifacts(directory: Path | None = None) -> dict:
    """Загружает модели. Бросает FileNotFoundError с понятным текстом, если файлов нет."""
    d = Path(directory) if directory else models_dir()
    paths = {"price": d / "price_model.joblib", "nlp": d / "room_type_clf.joblib"}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Не найдены файлы моделей в {d}: {', '.join(missing)}. "
            "Положите joblib-файлы из архива ноутбука в папку models/.")
    return {k: joblib.load(p) for k, p in paths.items()}


# ---------- текст ----------
def mask_contacts(text) -> str:
    """Маскирует e-mail, ссылки, @-имена и телефоны (как в ноутбуке)."""
    t = re.sub(r"<[^>]+>", " ", str(text))
    t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", t)
    t = re.sub(r"(?:https?://|www\.)\S+", "[URL]", t)
    t = re.sub(r"(?<!\w)@\w{3,}", "[HANDLE]", t)
    t = re.sub(r"(?<![\w])\+?\d[\d\s().-]{7,}\d(?![\w])", "[PHONE]", t)
    return re.sub(r"\s+", " ", t).strip()


def classify_room_type(text, nlp: dict) -> dict:
    """Определяет тип жилья по описанию. Возвращает метку, уверенность и флаг «не уверен»."""
    clean = mask_contacts(text or "")
    out = {"label": None, "confidence": 0.0, "uncertain": True,
           "ml_room_type": None, "reason": None, "text_used": clean}
    if len(clean) < MIN_TEXT_LEN:
        out["reason"] = "Текст слишком короткий для определения типа жилья."
        return out
    pipe = nlp["clf"]
    vec = pipe.named_steps["tfidf"]
    tokens = vec.build_analyzer()(clean)
    known = sum(1 for t in tokens if t in vec.vocabulary_)
    if known < MIN_KNOWN_TOKENS:
        out["reason"] = ("Текст не похож на описания, на которых обучена модель "
                         "(она обучена на английских описаниях Inside Airbnb).")
        return out
    proba = pipe.predict_proba([clean])[0]
    i = int(np.argmax(proba))
    label = str(pipe.classes_[i])
    conf = float(proba[i])
    out.update(label=label, confidence=conf)
    if conf >= nlp["threshold"]:
        out["uncertain"] = False
        out["ml_room_type"] = nlp["to_ml"].get(label)
    else:
        out["reason"] = f"Уверенность {conf:.0%} ниже порога {nlp['threshold']:.0%}."
    return out


# ---------- цена ----------
def parse_baths(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)", str(text))
    if m:
        return float(m.group(1))
    return 0.5 if "half" in str(text).lower() else np.nan


def validate_inputs(inp: dict, price_art: dict) -> list[str]:
    """Возвращает список понятных ошибок. Пустой список = вход корректен."""
    errors = []

    def num(key, label):
        v = inp.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            errors.append(f"{label}: нужно число.")
            return None

    for key, label in [("accommodates", "Гостей"), ("bedrooms", "Спален"),
                       ("baths", "Ванных"), ("n_amenities", "Число удобств")]:
        v = num(key, label)
        if v is None:
            if key in ("accommodates", "bedrooms", "baths"):
                errors.append(f"{label}: значение не указано.")
            continue
        lo, hi = LIMITS[key]
        if not (lo <= v <= hi):
            errors.append(f"{label}: допустимо от {lo} до {hi}, получено {v:g}.")
    rs = num("review_score", "Рейтинг")
    if rs is not None and not (LIMITS["review_score"][0] <= rs <= LIMITS["review_score"][1]):
        errors.append(f"Рейтинг: допустимо от 0 до 5, получено {rs:g}.")
    for key, label in [("latitude", "Широта"), ("longitude", "Долгота")]:
        v = num(key, label)
        if v is not None and not (LIMITS[key][0] <= v <= LIMITS[key][1]):
            errors.append(f"{label}: вне границ Стамбула ({LIMITS[key][0]}..{LIMITS[key][1]}).")
    districts = known_districts(price_art)
    if inp.get("neighbourhood") not in districts:
        errors.append("Район: выберите район из списка.")
    return errors


def known_districts(price_art: dict) -> list[str]:
    return sorted(price_art["district_median"].index.tolist())


def build_features(inp: dict, room_type: str | None, price_art: dict) -> pd.DataFrame:
    """Строит одну строку признаков в том же формате, что при обучении."""
    cat, num = price_art["cat"], price_art["num"]
    med = price_art["median_fill"]
    row = {c: np.nan for c in cat + num}
    row["room_type"] = room_type if room_type else np.nan
    row["neighbourhood_cleansed"] = inp["neighbourhood"]
    row["accommodates"] = float(inp["accommodates"])
    row["bedrooms"] = float(inp["bedrooms"])
    row["baths"] = float(inp["baths"])
    rs = inp.get("review_score")
    row["no_rating"] = 1 if rs is None or (isinstance(rs, float) and np.isnan(rs)) else 0
    row["review_scores_rating"] = float(rs) if row["no_rating"] == 0 else float(med["review_scores_rating"])
    cent = (price_art.get("district_centroids") or {}).get(inp["neighbourhood"])
    lat, lon = inp.get("latitude"), inp.get("longitude")
    row["latitude"] = float(lat) if lat is not None else (cent[0] if cent else float(med["latitude"]))
    row["longitude"] = float(lon) if lon is not None else (cent[1] if cent else float(med["longitude"]))
    row["n_amenities"] = float(inp.get("n_amenities", med["n_amenities"]))
    row["shared_bath"] = 1 if inp.get("shared_bath") else 0
    row["superhost"] = 1 if inp.get("superhost") else 0
    chosen = set(inp.get("amenities") or [])
    for i, name in enumerate(price_art["top_amenities"]):
        row[f"am_{i}"] = 1 if name in chosen else 0
    df = pd.DataFrame([row])[cat + num]
    df[cat] = df[cat].astype(object)  # пропуск в категории должен быть object, как при обучении
    return df


def predict_price(inp: dict, room_type: str | None, price_art: dict) -> dict:
    """Оценка цены за ночь в лирах: точка, диапазон и baseline (медиана по району)."""
    errors = validate_inputs(inp, price_art)
    if errors:
        raise ValueError("; ".join(errors))
    X = build_features(inp, room_type, price_art)
    price = float(np.expm1(price_art["model"].predict(X))[0])
    base = float(price_art["district_median"].get(inp["neighbourhood"], price_art["global_median"]))
    return {"price": price, "low": price * price_art["range_q10"],
            "high": price * price_art["range_q90"], "baseline": base,
            "room_type_used": room_type}
