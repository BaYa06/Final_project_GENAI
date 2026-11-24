"""Streamlit-приложение: помощник путешественника по Стамбулу.

Три раздела по задачам путешественника:
  1. «Проверить жильё»: параметры и описание объявления -> тип жилья (NLP) -> справедливая цена (ML)
     -> что важно знать по документам (RAG).
  2. «Вопросы о поездке»: свободные вопросы по официальным документам с источником.
  3. «О проекте»: данные, качество моделей, ограничения.

Запуск: streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl  # noqa: E402
import rag  # noqa: E402

st.set_page_config(page_title="Помощник по Стамбулу", page_icon="🕌", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .hero h1 {margin-bottom: 0.2rem;}
    .hero p {color: #666; font-size: 1.05rem; margin-top: 0;}
    div[data-testid="stForm"] {border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

ROOM_AUTO = "Определить по описанию"
ROOM_CHOICES = {"Entire home/apt": "Квартира или дом целиком", "Hotel room": "Номер в отеле",
                "Private room": "Отдельная комната"}
# Вопросы по документам, которые показываются под оценкой цены
PRE_BOOKING = [
    ("Нужно ли хозяину разрешение на посуточную аренду?",
     "Нужно ли хозяину квартиры разрешение на посуточную аренду?"),
    ("Кто отвечает за регистрацию гостей?",
     "Кто отвечает за уведомление о личности проживающих в жилье с разрешением на туристическую аренду?"),
    ("Какой налог на проживание?", "Какой налог на проживание в отеле?"),
]
EXAMPLES = [
    "Как купить İstanbulkart и где её пополнить?",
    "Сколько стоит проезд в метро Стамбула?",
    "Что делать, если потерял паспорт в Стамбуле?",
    "Нужна ли виза для граждан Кыргызстана?",
    "Что такое Tax Free и как вернуть НДС?",
    "How do I call an ambulance in Istanbul?",
]


@st.cache_resource
def get_artifacts():
    return pl.load_artifacts()


@st.cache_resource
def get_rag_index():
    return rag.load_index()


@st.cache_data
def get_metrics():
    path = pl.models_dir() / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_models():
    try:
        return get_artifacts(), None
    except FileNotFoundError as e:
        return None, str(e)


def ask_docs(question: str):
    """Возвращает (Answer | None, текст ошибки | None)."""
    try:
        return rag.answer(question, get_rag_index()), None
    except (FileNotFoundError, ValueError) as e:
        return None, f"Документы для ответов недоступны: {e}"


def show_answer(res: "rag.Answer"):
    if res.status in ("answered", "related"):
        if res.status == "related":
            st.warning("Прямого ответа в документах нет. Показан ближайший по теме фрагмент, "
                       "проверьте, отвечает ли он на ваш вопрос.")
            res_text = res.text.split("\n\n", 1)[-1]
        else:
            res_text = res.text
        st.markdown(res_text)
        st.caption("**Источники**")
        for s in res.sources:
            st.caption(f"• {s['source_name']}: [{s['title']}]({s['url']}), проверено {s['date']}")
    elif res.status == "refused":
        st.warning(res.text)
    else:
        st.info(res.text)


def show_nlp(res: dict):
    if res["label"] is None:
        st.warning(res["reason"])
    elif res["uncertain"]:
        st.warning(f"Не уверен: похоже на «{res['label']}» ({res['confidence']:.0%}). {res['reason']}")
    else:
        st.success(f"По описанию это **{res['label']}** (уверенность {res['confidence']:.0%}).")


def show_price(res: dict, asking: float | None):
    c1, c2, c3 = st.columns(3)
    c1.metric("Оценка за ночь", f"{res['price']:,.0f} ₺")
    c2.metric("Обычный диапазон", f"{res['low']:,.0f} – {res['high']:,.0f} ₺",
              help="Примерно 80% похожих объявлений попадают в этот диапазон.")
    c3.metric("Медиана по району", f"{res['baseline']:,.0f} ₺",
              help="Простое сравнение: типичная цена в районе без учёта параметров.")
    if asking:
        ratio = asking / res["price"]
        if asking < res["low"]:
            st.info(f"Цена в объявлении ({asking:,.0f} ₺) ниже обычного диапазона, это {ratio:.0%} от оценки. "
                    "Проверьте, что входит в цену и нет ли скрытых сборов.")
        elif asking > res["high"]:
            st.warning(f"Цена в объявлении ({asking:,.0f} ₺) выше обычного диапазона, это {ratio:.0%} от оценки. "
                       "Возможно, есть причины (вид, удобства), но стоит сравнить с похожими вариантами.")
        else:
            st.success(f"Цена в объявлении ({asking:,.0f} ₺) в обычном диапазоне, это {ratio:.0%} от оценки.")
    st.caption("Цены из котировок Airbnb в турецких лирах, в основном на даты сентября–октября 2026. "
               "Оценка ориентировочная. Приложение не определяет мошенничество.")


def summary_text(nlp_res, room_choice, room_type, price_res, asking, question, q_res) -> str:
    """Короткий итог одним сообщением: тип жилья, цена, ответ на вопрос."""
    lines = []
    if nlp_res is not None and nlp_res["label"] and not nlp_res["uncertain"]:
        lines.append(f"- **Тип жилья:** {nlp_res['label']} (уверенность {nlp_res['confidence']:.0%}).")
    elif nlp_res is not None:
        lines.append("- **Тип жилья:** по описанию определить не удалось, цена посчитана без него.")
    elif room_type:
        lines.append(f"- **Тип жилья:** {ROOM_CHOICES.get(room_type, room_type)} (указан вручную).")
    if price_res:
        lines.append(f"- **Цена за ночь:** около {price_res['price']:,.0f} ₺ "
                     f"(обычный диапазон {price_res['low']:,.0f} – {price_res['high']:,.0f} ₺).")
        if asking:
            if asking < price_res["low"]:
                verdict = "ниже обычного диапазона, проверьте, что входит в цену"
            elif asking > price_res["high"]:
                verdict = "выше обычного диапазона, сравните с похожими вариантами"
            else:
                verdict = "в обычном диапазоне"
            lines.append(f"- **Цена в объявлении** {asking:,.0f} ₺: {verdict}.")
    else:
        lines.append("- **Цена:** не удалось посчитать, проверьте введённые параметры.")
    if question.strip() and q_res is not None:
        if q_res.status == "answered":
            lines.append("- **Ваш вопрос:** ответ найден в документах, он ниже, с источником.")
        elif q_res.status == "related":
            lines.append("- **Ваш вопрос:** прямого ответа в документах нет, ниже ближайший фрагмент.")
        elif q_res.status == "refused":
            lines.append("- **Ваш вопрос:** помощник отвечает только по документам для приезжих.")
        else:
            lines.append("- **Ваш вопрос:** в документах ответа нет (не знаю).")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>🕌 Помощник по Стамбулу</h1>'
    "<p>Проверьте цену жилья за ночь и получите ответы по официальным документам: виза, транспорт, "
    "аренда, безопасность, налоги.</p></div>",
    unsafe_allow_html=True,
)

tab_listing, tab_ask, tab_map, tab_about = st.tabs(
    ["🏠 Проверить жильё", "💬 Вопросы о поездке", "🗺️ Цены по районам", "ℹ️ О проекте"])

art, models_error = load_models()

# ---------------------------------- 1. Проверить жильё ---------------------------------------
with tab_listing:
    st.markdown("Заполните данные из объявления и нажмите кнопку внизу. Вы получите оценку цены за ночь "
                "и подсказки по правилам аренды.")
    if models_error:
        st.error(models_error)
    else:
        price_art, nlp_art = art["price"], art["nlp"]
        with st.form("listing_form"):
            st.markdown("##### 1. Параметры жилья")
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            districts = pl.known_districts(price_art)
            neighbourhood = c1.selectbox(
                "Район", districts, index=districts.index("Beyoglu") if "Beyoglu" in districts else 0)
            accommodates = c2.number_input("Гостей", value=2, step=1)
            bedrooms = c3.number_input("Спален", value=1, step=1, help="0 = студия")
            baths = c4.number_input("Ванных", value=1.0, step=0.5)

            st.markdown("##### 2. Описание из объявления (по желанию)")
            text = st.text_area("Вставьте текст описания, лучше на английском", height=130,
                                label_visibility="collapsed",
                                placeholder="Например: Spacious apartment with a balcony and a fully "
                                            "equipped kitchen near the metro...")
            room_choice = st.selectbox(
                "Тип жилья", [ROOM_AUTO] + list(ROOM_CHOICES) + ["Не указывать"],
                format_func=lambda k: ROOM_CHOICES.get(k, k),
                help="По умолчанию тип определяется по описанию. Можно указать вручную.")

            user_question = st.text_input(
                "Ваш вопрос по документам (по желанию)",
                placeholder="Например: Что нужно для въезда в Турцию гражданину Кыргызстана?")

            with st.expander("Дополнительно: цена из объявления, рейтинг, удобства"):
                a1, a2, a3 = st.columns(3)
                asking = a1.number_input("Цена в объявлении, ₺ за ночь", value=0, step=100,
                                         help="Оставьте 0, если не хотите сравнивать.")
                rating = a2.number_input("Рейтинг (0–5)", value=0.0, step=0.1,
                                         help="Оставьте 0, если у жилья пока нет оценок.")
                n_am = a3.number_input("Число удобств в объявлении",
                                       value=int(price_art["median_fill"]["n_amenities"]), step=1)
                b1, b2 = st.columns(2)
                shared_bath = b1.checkbox("Общая ванная")
                superhost = b2.checkbox("Хозяин — суперхост")
                amenities = st.multiselect("Частые удобства", price_art["top_amenities"])

            submitted = st.form_submit_button("Оценить цену", type="primary", use_container_width=True)

        if submitted:
            inputs = {"neighbourhood": neighbourhood, "accommodates": accommodates, "bedrooms": bedrooms,
                      "baths": baths, "shared_bath": shared_bath, "superhost": superhost,
                      "review_score": rating if rating > 0 else None, "n_amenities": n_am,
                      "amenities": amenities}
            asking_price = asking if asking > 0 else None
            st.divider()
            st.subheader("Результат")

            # Шаг 1. NLP: тип жилья
            nlp_res, room_type = None, None
            if room_choice == ROOM_AUTO:
                if text.strip():
                    nlp_res = pl.classify_room_type(text, nlp_art)
                    room_type = nlp_res["ml_room_type"]
            elif room_choice in ROOM_CHOICES:
                room_type = room_choice

            # Шаг 2. ML: цена
            price_res, price_error = None, None
            try:
                price_res = pl.predict_price(inputs, room_type, price_art)
            except ValueError as e:
                price_error = str(e)

            # Шаг 3. RAG: вопрос пользователя
            q_res, q_err = (ask_docs(user_question) if user_question.strip() else (None, None))

            # Итог: всё в одном сообщении
            with st.container(border=True):
                st.markdown("#### Итог")
                st.markdown(summary_text(nlp_res, room_choice, room_type, price_res, asking_price,
                                         user_question, q_res))
            if price_error:
                st.error(f"Проверьте ввод: {price_error}")

            st.markdown("##### Подробности")
            if nlp_res is not None:
                st.markdown("**Что понятно из описания**")
                show_nlp(nlp_res)
                if room_type is None:
                    st.caption("Тип жилья не определён, поэтому цена считается без него.")
            elif room_choice == ROOM_AUTO:
                st.caption("Описание не введено: тип жилья не используется в расчёте цены.")

            if price_res:
                st.markdown("**Оценка цены**")
                show_price(price_res, asking_price)
                with st.expander("Что влияет на цену (что-если)"):
                    rows = pl.explain_price(inputs, room_type, price_art)
                    for r in rows:
                        sign = "+" if r["delta"] >= 0 else "−"
                        st.markdown(f"- {r['label']}: **{sign}{abs(r['delta']):,.0f} ₺** за ночь")
                    st.caption("Это чувствительность модели при прочих равных, а не причинная связь: "
                               "реальные параметры жилья связаны между собой (например, число спален и "
                               "гостей), поэтому отдельный шаг иногда выглядит неожиданно.")

            if user_question.strip():
                st.markdown("**Ответ на ваш вопрос**")
                if q_err:
                    st.error(q_err)
                else:
                    show_answer(q_res)

            st.markdown("**Что важно знать перед бронированием**")
            for title, question in PRE_BOOKING:
                with st.expander(title):
                    res, err = ask_docs(question)
                    if err:
                        st.error(err)
                    else:
                        show_answer(res)
            st.caption("Другие вопросы задайте во вкладке «Вопросы о поездке».")

# ---------------------------------- 2. Вопросы о поездке -------------------------------------
with tab_ask:
    st.markdown("Задайте вопрос о визе, транспорте, аренде, безопасности или налогах. Ответ собирается из "
                "официальных документов, внизу всегда указан источник. Если в документах ответа нет, "
                "помощник так и скажет.")
    st.caption("Примеры вопросов (нажмите, чтобы подставить):")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state["ask_q"] = ex
    question = st.text_input("Ваш вопрос", key="ask_q", placeholder="Например: Нужна ли виза в Турцию?")
    if question.strip():
        res, err = ask_docs(question)
        st.divider()
        if err:
            st.error(err)
        else:
            show_answer(res)

# ---------------------------------- 3. Цены по районам ----------------------------------------
with tab_map:
    st.markdown("Типичная цена за ночь по районам Стамбула (медиана по объявлениям в данных). "
                "Чем больше круг, тем выше цена. Это простое сравнение без учёта параметров жилья.")
    if models_error:
        st.error(models_error)
    else:
        import pandas as pd
        cents = art["price"].get("district_centroids") or {}
        med = art["price"]["district_median"]
        rows = [{"Район": d, "Медиана, ₺": float(med[d]), "lat": cents[d][0], "lon": cents[d][1]}
                for d in med.index if d in cents]
        if not rows:
            st.info("В модели нет координат районов.")
        else:
            dfm = pd.DataFrame(rows).sort_values("Медиана, ₺", ascending=False)
            st.map(dfm.assign(size=dfm["Медиана, ₺"] * 0.5), latitude="lat", longitude="lon", size="size")
            c1, c2 = st.columns(2)
            c1.markdown("**Самые дорогие районы**")
            c1.dataframe(dfm[["Район", "Медиана, ₺"]].head(8).round(0), hide_index=True)
            c2.markdown("**Самые доступные районы**")
            c2.dataframe(dfm[["Район", "Медиана, ₺"]].tail(8).iloc[::-1].round(0), hide_index=True)
            st.caption("Данные: Inside Airbnb, Стамбул, выпуск 25.09.2026 (CC BY 4.0). Это цены "
                       "объявлений, а не оценка безопасности или удобства района.")

# ---------------------------------- 4. О проекте ---------------------------------------------
with tab_about:
    st.subheader("Как это работает")
    st.markdown(
        "- **Тип жилья по тексту (NLP).** Модель читает описание объявления и определяет тип жилья. "
        "Если не уверена, так и говорит.\n"
        "- **Оценка цены (ML).** Градиентный бустинг считает цену за ночь по параметрам жилья и "
        "показывает обычный диапазон. Для сравнения есть простая медиана по району.\n"
        "- **Ответы по документам (RAG).** Поиск по 18 официальным документам. В ответе всегда есть "
        "источник и дата проверки. Вне документов отвечает «не знаю».")
    metrics = get_metrics()
    st.subheader("Качество (отложенная выборка, seed 42)")
    if metrics is None:
        st.info("Файл models/metrics.json не найден.")
    else:
        ml, nlp = metrics["ml"], metrics["nlp"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Ошибка цены (MAE)", f"{ml['mae_gb_final_true_room_type']:,.0f} ₺",
                  help="Средняя абсолютная ошибка модели на тесте.")
        m2.metric("Ошибка простого сравнения", f"{ml['mae_baseline_district_median']:,.0f} ₺",
                  help="Медиана по району. Модель должна быть заметно лучше.")
        nlp_with = nlp.get("with_threshold", {})
        if nlp_with:
            m3.metric("Тип жилья, macro-F1", f"{nlp_with['macro_f1']:.2f}",
                      help=f"При пороге уверенности {nlp['threshold']}: отвечает на "
                           f"{nlp_with['answered_share']:.0%} описаний.")
        st.caption(f"Обучающих объявлений: {metrics['data']['n_train']:,}, тестовых: "
                   f"{metrics['data']['n_test']:,}.")
        cv = ml.get("cv_5fold_train")
        if cv:
            st.caption(f"Кросс-валидация (5 фолдов): модель {cv['model_mae_mean']:,} ± {cv['model_mae_std']} ₺, "
                       f"простое сравнение {cv['baseline_mae_mean']:,} ± {cv['baseline_mae_std']} ₺.")
    st.subheader("Данные и ограничения")
    st.markdown(
        "- Данные: [Inside Airbnb](https://insideairbnb.com/get-the-data/), Стамбул, выпуск 25.09.2026, "
        "лицензия CC BY 4.0. Цены в турецких лирах, только короткие котировки.\n"
        "- Цены в основном на сентябрь–октябрь 2026, один выпуск данных. Объекты дороже 39 366 ₺ за ночь "
        "не оцениваются.\n"
        "- Определение типа жилья обучено на английских описаниях. На русском или турецком текст "
        "модель отвечает «не уверен».\n"
        "- Ответы по документам собираются из текста документов, а не пересказываются. Поиск идёт по "
        "словам, редкие формулировки вопроса могут дать «не знаю».\n"
        "- Приложение не определяет мошенничество и не заменяет консультацию официальных служб.")
    st.subheader("Команда")
    st.markdown("Kholikov Baiirbek (ML и приложение), Muratbekov Batyi (RAG), Amatjanov Nurbek (NLP). "
                "Преподаватель: Наргиза Жумалиева.")
