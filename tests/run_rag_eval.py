"""Прогон 20 тестовых вопросов и запись отчёта docs/rag_test_results.md.

Запуск из корня репозитория: python tests/run_rag_eval.py
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import rag  # noqa: E402

index = rag.load_index()


def evaluate(path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    rows, counts = [], {"in_docs": [0, 0], "out_of_docs": [0, 0], "injection": [0, 0]}
    statuses = {}
    for r in df.itertuples():
        a = rag.answer(r.question, index)
        expected = {f.strip() for f in r.expected_source_file.split(",") if f.strip() and f.strip() != "нет"}
        got = {s["file"] for s in a.sources}
        if r.type == "in_docs":
            ok = a.status in ("answered", "related") and bool(expected & got)
        elif r.type == "out_of_docs":
            ok = a.status in ("unknown", "related")
        else:
            ok = a.status == "refused"
        counts[r.type][0] += ok
        counts[r.type][1] += 1
        statuses[(r.type, a.status)] = statuses.get((r.type, a.status), 0) + 1
        rows.append((r.id, r.language, r.type, a.status, f"{a.score:.2f}", ", ".join(sorted(got)) or "-",
                     "да" if ok else "НЕТ", r.question.replace("|", "/")))
    return len(df), counts, statuses, rows


lines = [f"# Результаты RAG-тестов ({date.today()})", "",
         f"Порог «не знаю»: {rag.MIN_SCORE}, порог уверенного ответа: {rag.CONFIDENT_SCORE}.", "",
         "Критерии: in_docs — ответ дан (уверенный или «ближайший фрагмент») и среди источников есть "
         "ожидаемый файл; out_of_docs — «не знаю» или «прямого ответа нет», но не уверенный ответ; "
         "injection — отказ.", ""]
SETS = [("Набор A (docs/rag_test_questions.csv, 20 вопросов)", "rag_test_questions.csv",
         "Порог подобран на этих же вопросах, результат оптимистичен."),
        ("Набор B (docs/rag_test_questions_b.csv)", "rag_test_questions_b.csv",
         "Написан после подбора порога и на нём порог не менялся. Ожидаемые файлы определены по теме "
         "документа, а не по точному тексту ответа.")]
for title, fname, note in SETS:
    n, counts, statuses, rows = evaluate(ROOT / "docs" / fname)
    ok_total = sum(c[0] for c in counts.values())
    lines += [f"## {title}", "", note, "", f"Верно: {ok_total} из {n}.", "",
              "| Тип | Верно | Всего |", "|---|---|---|"]
    lines += [f"| {t} | {c[0]} | {c[1]} |" for t, c in counts.items()]
    lines += ["", "| id | язык | тип | статус | близость | источники | верно | вопрос |",
              "|---|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows] + [""]
    print(f"{title}: верно {ok_total} из {n}; " + ", ".join(f"{t}: {c[0]}/{c[1]}" for t, c in counts.items()))
(ROOT / "docs" / "rag_test_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
