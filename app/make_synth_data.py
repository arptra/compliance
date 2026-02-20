from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_MONTHS = ["2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11"]
DEC_MONTH = "2025-12"


def make_messages(rng: np.random.Generator, n: int = 1200) -> pd.DataFrame:
    neutral = [
        "Подскажите как открыть вклад в приложении",
        "Спасибо за быстрый ответ",
        "Уточните лимиты по карте",
        "Как сменить номер телефона",
        "Вопрос по условиям тарифа",
    ]
    complaints = [
        "Не работает перевод между счетами",
        "Списали деньги без предупреждения",
        "Верните возврат по ошибочной операции",
        "Это ужасно, не могу войти в личный кабинет",
        "Жалоба на работу поддержки, обман",
    ]
    december_novel = [
        "Не работает QR-оплата в метро, не могу заплатить",
        "Пропали бонусы за подписку кино",
        "Мошенничество через NFC метку, срочно верните деньги",
        "Жалоба: биометрия в банкомате отклоняет лицо",
    ]

    rows = []
    for m in BASELINE_MONTHS:
        for _ in range(n // len(BASELINE_MONTHS)):
            text = rng.choice(complaints if rng.random() < 0.35 else neutral)
            day = int(rng.integers(1, 28))
            rows.append({"date": f"{m}-{day:02d}", "message": text})

    for _ in range(max(200, n // 6)):
        bucket = rng.random()
        if bucket < 0.25:
            text = rng.choice(december_novel)
        elif bucket < 0.6:
            text = rng.choice(complaints)
        else:
            text = rng.choice(neutral)
        day = int(rng.integers(1, 28))
        rows.append({"date": f"{DEC_MONTH}-{day:02d}", "message": text})

    return pd.DataFrame(rows)


def main(out: str) -> None:
    rng = np.random.default_rng(42)
    df = make_messages(rng)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"Wrote synthetic dataset to {out_path} with {len(df)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sample.xlsx")
    args = parser.parse_args()
    main(args.out)
