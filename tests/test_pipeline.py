import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import pipeline as pl  # noqa: E402

AM = [f"amenity {i}" for i in range(15)]
BASE_NUM = ["accommodates", "bedrooms", "baths", "review_scores_rating", "latitude", "longitude",
            "n_amenities", "shared_bath", "superhost", "no_rating"]


def make_price_art():
    rng = np.random.default_rng(42)
    n = 300
    cat = ["room_type", "neighbourhood_cleansed"]
    num = BASE_NUM + [f"am_{i}" for i in range(15)]
    df = pd.DataFrame({
        "room_type": rng.choice(["Entire home/apt", "Private room", "Hotel room"], n),
        "neighbourhood_cleansed": rng.choice(["Beyoglu", "Kadikoy"], n)})
    for c in num:
        df[c] = rng.random(n)
    y = np.log1p(3000 + 2000 * df["accommodates"] + 500 * rng.random(n))
    pre = ColumnTransformer([
        ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                             encoded_missing_value=np.nan), cat),
        ("n", "passthrough", num)])
    model = Pipeline([("p", pre), ("m", HistGradientBoostingRegressor(
        categorical_features=[True, True] + [False] * len(num), max_iter=30, random_state=42))])
    model.fit(df[cat + num], y)
    return {"model": model, "cat": cat, "num": num,
            "median_fill": pd.Series({c: 0.5 for c in num}),
            "top_amenities": AM, "range_q10": 0.7, "range_q90": 1.5, "price_cap_try": 40000.0,
            "district_median": pd.Series({"Beyoglu": 4500.0, "Kadikoy": 4000.0}),
            "global_median": 4200.0,
            "district_centroids": {"Beyoglu": (41.03, 28.98), "Kadikoy": (40.99, 29.03)}}


def make_nlp_art():
    texts = ["entire apartment with kitchen and balcony near metro",
             "private room in shared flat with friendly host",
             "hotel room with breakfast and daily cleaning service",
             "whole flat sea view terrace modern kitchen",
             "cozy private bedroom in family home",
             "hotel suite reception 24 hour front desk"] * 5
    labels = ["Entire home/apt", "Private/Shared room", "Hotel room"] * 10
    labels = (["Entire home/apt", "Private/Shared room", "Hotel room",
               "Entire home/apt", "Private/Shared room", "Hotel room"] * 5)
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                     ("lr", LogisticRegression(max_iter=500, random_state=42))])
    pipe.fit(texts, labels)
    return {"clf": pipe, "threshold": 0.4, "classes": list(pipe.classes_),
            "to_ml": {"Entire home/apt": "Entire home/apt", "Hotel room": "Hotel room",
                      "Private/Shared room": "Private room"}}


GOOD = {"neighbourhood": "Beyoglu", "accommodates": 2, "bedrooms": 1, "baths": 1.0,
        "shared_bath": False, "superhost": False, "review_score": None,
        "n_amenities": 30, "amenities": ["amenity 1"]}


def test_mask_contacts():
    t = pl.mask_contacts("Call +90 532 123 45 67 or mail me a@b.com, see https://x.com/y and @host_name")
    assert "[PHONE]" in t and "[EMAIL]" in t and "[URL]" in t and "[HANDLE]" in t
    assert "532" not in t and "a@b.com" not in t


def test_parse_baths():
    assert pl.parse_baths("1.5 baths") == 1.5
    assert pl.parse_baths("Half-bath") == 0.5
    assert np.isnan(pl.parse_baths(None))


@pytest.mark.parametrize("key,value", [("accommodates", -5), ("accommodates", 0), ("bedrooms", -1),
                                       ("baths", 99), ("review_score", 7), ("latitude", 10.0)])
def test_validation_rejects_impossible_values(key, value):
    art = make_price_art()
    bad = {**GOOD, key: value}
    assert pl.validate_inputs(bad, art)
    with pytest.raises(ValueError):
        pl.predict_price(bad, None, art)


def test_validation_unknown_district_and_missing():
    art = make_price_art()
    assert pl.validate_inputs({**GOOD, "neighbourhood": "Nowhere"}, art)
    assert pl.validate_inputs({**GOOD, "bedrooms": None}, art)


def test_predict_price_range_ordering():
    art = make_price_art()
    res = pl.predict_price(GOOD, "Entire home/apt", art)
    assert res["low"] < res["price"] < res["high"]
    assert res["baseline"] == 4500.0


def test_predict_price_with_missing_room_type():
    art = make_price_art()
    res = pl.predict_price(GOOD, None, art)
    assert res["price"] > 0


def test_features_have_training_columns():
    art = make_price_art()
    X = pl.build_features(GOOD, "Hotel room", art)
    assert list(X.columns) == art["cat"] + art["num"]
    assert X.loc[0, "no_rating"] == 1 and X.loc[0, "am_1"] == 1


def test_classify_short_text_is_uncertain():
    r = pl.classify_room_type("hi", make_nlp_art())
    assert r["uncertain"] and r["label"] is None and r["reason"]


def test_classify_unknown_language_is_uncertain():
    r = pl.classify_room_type("Просторная квартира в центре города рядом с метро и парком", make_nlp_art())
    assert r["uncertain"] and r["label"] is None


def test_classify_normal_text_returns_label():
    r = pl.classify_room_type("entire apartment with kitchen and balcony near metro", make_nlp_art())
    assert r["label"] in {"Entire home/apt", "Private/Shared room", "Hotel room"}
    assert 0.0 <= r["confidence"] <= 1.0


def test_classify_masks_contacts():
    r = pl.classify_room_type("entire apartment kitchen balcony call +90 532 123 45 67 now", make_nlp_art())
    assert "532" not in r["text_used"]


def test_load_artifacts_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        pl.load_artifacts(tmp_path)
