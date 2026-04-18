#!/usr/bin/env python3
"""
Загрузка сохранённого пайплайна train_neural_tuned_full.py и предсказание status.

Вход: CSV с теми же столбцами, что и full_features_dataset.csv, **без** столбца
status (если status есть — он игнорируется). Можно одну или несколько строк.

Интерактивно запрашивается путь к CSV; либо передайте --csv.

Примеры:
  python predict_neural_mlp.py
  python predict_neural_mlp.py --csv /path/to/rows.csv
  python predict_neural_mlp.py --model models/neural_tabular_mlp.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "models" / "neural_tabular_mlp.joblib"


def _expected_feature_columns(pipe) -> list[str]:
    if hasattr(pipe, "feature_names_in_") and pipe.feature_names_in_ is not None:
        return [str(c) for c in pipe.feature_names_in_]
    prep = pipe.named_steps.get("prep")
    if prep is not None and getattr(prep, "feature_names_in_", None) is not None:
        return [str(c) for c in prep.feature_names_in_]
    raise RuntimeError(
        "В сохранённой модели нет feature_names_in_. "
        "Переобучите train_neural_tuned_full.py на pandas DataFrame."
    )


def _prepare_X(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    if "status" in df.columns:
        df = df.drop(columns=["status"])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"В CSV не хватает столбцов: {missing}")
    extra = [c for c in df.columns if c not in expected]
    if extra:
        print(f"Предупреждение: лишние столбцы будут отброшены: {extra}", file=sys.stderr)
    return df[expected].copy()


def run_predict(pipe, X: pd.DataFrame) -> None:
    proba = pipe.predict_proba(X)
    pred = pipe.predict(X)
    p_phish = proba[:, 1]
    for i in range(len(X)):
        cls = int(pred[i])
        label = "фишинг (1)" if cls == 1 else "не фишинг (0)"
        print(
            f"  строка {i + 1}: класс={cls} ({label}), "
            f"P(фишинг)={float(p_phish[i]):.6f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Предсказание нейросетевой MLP по CSV признаков.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Путь к .joblib от train_neural_tuned_full.py",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV с признаками; если не задан — запрос в консоли",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(
            f"Нет файла модели: {args.model}\n"
            "Сначала обучите и сохраните: python train_neural_tuned_full.py"
        )

    pipe = joblib.load(args.model)
    expected = _expected_feature_columns(pipe)

    print("Загружена модель:", args.model.resolve(), flush=True)
    print("Ожидаемые столбцы (как в full_features без status):", ", ".join(expected), flush=True)

    def process_csv(csv_path: Path) -> bool:
        if not csv_path.is_file():
            print(f"Файл не найден: {csv_path}", flush=True)
            return False
        df = pd.read_csv(csv_path)
        try:
            X = _prepare_X(df, expected)
        except ValueError as e:
            print(e, flush=True)
            return False
        print(f"\nФайл: {csv_path} | строк: {len(X)}", flush=True)
        run_predict(pipe, X)
        return True

    if args.csv is not None:
        if not process_csv(args.csv):
            raise SystemExit(1)
        return

    while True:
        try:
            line = input("\nПуть к CSV с признаками (пусто — выход): ").strip()
        except EOFError:
            break
        if not line:
            break
        process_csv(Path(line).expanduser())


if __name__ == "__main__":
    main()
