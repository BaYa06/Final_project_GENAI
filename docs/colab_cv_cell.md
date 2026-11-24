# Кросс-валидация MAE (ячейка 13 ноутбука)

Сейчас сравнение вариантов (baseline, Ridge, бустинг, с типом и без) сделано на одном разбиении 80/20. Эта ячейка считает MAE по 5 фолдам и показывает среднее и разброс. Ячейка уже добавлена в ноутбук (ячейка 13) и запущена локально после всех остальных ячеек. Результат: бустинг MAE 1 478 ± 26 ₺, baseline 2 585 ± 45 ₺ (по фолдам: 1459, 1515, 1456, 1504, 1453 и 2593, 2665, 2574, 2531, 2561). Числа записаны в `models/metrics.json` (ключ `ml.cv_5fold_train`). Расчёт идёт по обучающей выборке, тест не используется.

```python
# Ячейка 13. Кросс-валидация MAE (5 фолдов, seed 42)
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cols = c_np + num   # признаки финальной модели (c_np задан в ячейке 11)
X, y = train[cols], ytr   # ytr — массив цен train (numpy)

maes_model, maes_base = [], []
for tr, va in kf.split(X):
    m = clone(g_np).fit(X.iloc[tr], np.log1p(y[tr]))
    pred = np.expm1(m.predict(X.iloc[va]))
    maes_model.append(mean_absolute_error(y[va], pred))
    med = train.iloc[tr].groupby("neighbourhood_cleansed")["price_try"].median()
    base = train.iloc[va]["neighbourhood_cleansed"].map(med).fillna(np.median(y[tr]))
    maes_base.append(mean_absolute_error(y[va], base))

for name, v in [("Бустинг", maes_model), ("Baseline: медиана по району", maes_base)]:
    print(f"{name:30s} MAE = {np.mean(v):7.0f} ± {np.std(v):4.0f} ₺ (по 5 фолдам: {np.round(v)})")
```


