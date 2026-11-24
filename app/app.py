"""Streamlit-приложение: помощник путешественника по Стамбулу.

Запуск: streamlit run app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl  # noqa: E402

st.set_page_config(page_title="Istanbul Traveller Assistant", page_icon="🕌", layout="wide")


@st.cache_resource
def get_artifacts():
    return pl.load_artifacts()


st.title("Помощник путешественника по Стамбулу")
st.caption("Оценка цены за ночь (₺), чтение описания объявления и ответы по документам. "
           "Данные: Inside Airbnb, Стамбул, выпуск 25.09.2026 (CC BY 4.0).")

try:
    art = get_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

price_art, nlp_art = art["price"], art["nlp"]
districts = pl.known_districts(price_art)

with st.sidebar:
    st.header("Параметры жилья")
    neighbourhood = st.selectbox("Район", districts)
    accommodates = st.number_input("Гостей", value=2, step=1)
    bedrooms = st.number_input("Спален (0 = студия)", value=1, step=1)
    baths = st.number_input("Ванных", value=1.0, step=0.5)
    shared_bath = st.checkbox("Общая ванная")
    superhost = st.checkbox("Суперхост")
    use_rating = st.checkbox("Указать рейтинг", value=False)
    rating = st.number_input("Рейтинг (0–5)", value=4.7, step=0.1) if use_rating else None
    n_am = st.number_input("Число удобств в объявлении", value=int(price_art["median_fill"]["n_amenities"]), step=1)
    amenities = st.multiselect("Отметьте удобства из списка частых", price_art["top_amenities"])

inputs = {"neighbourhood": neighbourhood, "accommodates": accommodates, "bedrooms": bedrooms,
          "baths": baths, "shared_bath": shared_bath, "superhost": superhost,
          "review_score": rating, "n_amenities": n_am, "amenities": amenities}


def show_price(res: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("Оценка цены за ночь", f"{res['price']:,.0f} ₺")
    c2.metric("Диапазон (около 80% случаев)", f"{res['low']:,.0f} – {res['high']:,.0f} ₺")
    c3.metric("Baseline: медиана по району", f"{res['baseline']:,.0f} ₺")
    st.caption("Цены из котировок Airbnb в турецких лирах, в основном на даты сентября–октября 2026. "
               "Приложение не определяет мошенничество.")


def show_nlp(res: dict):
    if res["label"] is None:
        st.warning(res["reason"])
    elif res["uncertain"]:
        st.warning(f"Не уверен. Лучший вариант: {res['label']} ({res['confidence']:.0%}). {res['reason']}")
    else:
        st.success(f"Тип жилья: {res['label']} (уверенность {res['confidence']:.0%})")


tab1, tab2, tab3 = st.tabs(["Оценка цены (ML)", "Тип жилья по тексту (NLP)", "Помощник"])

with tab1:
    st.subheader("Оценка цены по параметрам")
    rt_choice = st.selectbox("Тип жилья", ["Не указывать", "Entire home/apt", "Hotel room", "Private room"])
    if st.button("Оценить цену", key="price_btn"):
        try:
            res = pl.predict_price(inputs, None if rt_choice == "Не указывать" else rt_choice, price_art)
            show_price(res)
        except ValueError as e:
            st.error(f"Проверьте ввод: {e}")

with tab2:
    st.subheader("Определение типа жилья по описанию")
    text = st.text_area("Описание объявления (лучше на английском)", height=180, key="nlp_text")
    if st.button("Определить тип", key="nlp_btn"):
        show_nlp(pl.classify_room_type(text, nlp_art))
    st.caption("Телефоны, e-mail и ссылки в тексте маскируются до анализа.")

with tab3:
    st.subheader("Сквозной запрос: текст объявления → тип жилья → цена")
    text3 = st.text_area("Вставьте описание объявления", height=160, key="full_text")
    question = st.text_input("Ваш вопрос (например: справедлива ли цена? что нужно для въезда?)")
    if st.button("Спросить", key="full_btn"):
        nlp_res = pl.classify_room_type(text3, nlp_art)
        st.markdown("**1. NLP читает объявление**")
        show_nlp(nlp_res)
        st.markdown("**2. ML оценивает цену**")
        try:
            res = pl.predict_price(inputs, nlp_res["ml_room_type"], price_art)
            show_price(res)
            if nlp_res["ml_room_type"] is None:
                st.info("Тип жилья не определён, поэтому модель считает без него.")
        except ValueError as e:
            st.error(f"Проверьте ввод: {e}")
        st.markdown("**3. RAG отвечает по документам**")
        st.info("RAG-часть ещё не подключена: документы для приезжих собираются. "
                "Здесь появится ответ со ссылкой на источник или «не знаю».")
