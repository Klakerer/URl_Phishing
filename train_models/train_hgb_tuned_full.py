#!/usr/bin/env python3
"""
Финальное обучение HistGradientBoostingClassifier на полном датасете
с гиперпараметрами из успешного прогона tune_hgb.py (см. TUNED_HGB_KWARGS).

Данные: train_test_split со стратификацией — модель учится на train (все
остальные строки после отложения test), метрики считаются на test.

Пример:
  python train_hgb_tuned_full.py
  python train_hgb_tuned_full.py --test-size 0.2 --save-pipeline hgb_full.joblib
  python train_hgb_tuned_full.py --params-json hgb_search.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from train_classical_models import build_preprocessor, read_csv_with_progress

# Параметры из лучшего кандидата RandomizedSearchCV (терминал: tune_hgb.py)
TUNED_HGB_KWARGS: dict[str, Any] = {
    "early_stopping": False,
    "l2_regularization": 0.0016095289638234304,
    "learning_rate": 0.07654654942662646,
    "max_bins": 127,
    "max_depth": 9,
    "max_iter": 494,
    "min_samples_leaf": 148,
    "n_iter_no_change": 10,
    "validation_fraction": 0.05,
}


def load_hgb_kwargs_from_json(path: Path) -> dict[str, Any]:
    """Берёт best_params из JSON tune_hgb.py (--save-results) и мапит в kwargs HGB."""
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("best_params") or data
    if not isinstance(raw, dict):
        raise ValueError("В JSON ожидается объект с ключами model__...")
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if not k.startswith("model__"):
            continue
        name = k.removeprefix("model__")
        out[name] = v
    if not out:
        raise ValueError("Нет ключей model__* в JSON")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HGB с подобранными гиперпараметрами: полные данные, F1 на test.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/danya/datasets/andei_dataset"),
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default="full_features_dataset.csv",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--params-json",
        type=Path,
        default=None,
        help="JSON от tune_hgb.py (--save-results): взять best_params вместо TUNED_HGB_KWARGS",
    )
    parser.add_argument(
        "--save-pipeline",
        type=Path,
        default=None,
        help="Сохранить обученный Pipeline в .joblib",
    )
    args = parser.parse_args()

    features_path = args.data_dir / args.features_file
    if not features_path.is_file():
        raise SystemExit(f"Нет файла: {features_path}")

    if args.params_json is not None:
        if not args.params_json.is_file():
            raise SystemExit(f"Нет файла: {args.params_json}")
        hgb_kw = load_hgb_kwargs_from_json(args.params_json)
        hgb_kw.setdefault("random_state", args.random_state)
        print(f"Параметры HGB из {args.params_json}", flush=True)
    else:
        hgb_kw = {**TUNED_HGB_KWARGS, "random_state": args.random_state}

    print("=== HGB: полный датасет, подобранные гиперпараметры ===", flush=True)
    print(f"CSV: {features_path}", flush=True)

    df = read_csv_with_progress(features_path, nrows=None)
    if "status" not in df.columns:
        raise SystemExit("Нет столбца status.")
    if df["status"].nunique() < 2:
        raise SystemExit("Нужны оба класса status.")

    y = df["status"].astype(int).values
    X = df.drop(columns=["status"])
    print(f"Строк: {len(df)}, признаков: {X.shape[1]}", flush=True)
    print("Классы (весь файл):", dict(Counter(y)), flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    print(
        f"Train: {len(X_train)} | Test: {len(X_test)} | "
        f"test_size={args.test_size}",
        flush=True,
    )

    preprocessor = build_preprocessor(X_train)
    hgb = HistGradientBoostingClassifier(**hgb_kw)
    pipe = Pipeline([("prep", preprocessor), ("model", hgb)])

    print("\nОбучение (fit на train)...", flush=True)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    f1_bin = float(f1_score(y_test, y_pred, average="binary"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    auc = float(roc_auc_score(y_test, proba))

    print("\n========== Test (отложенная выборка) ==========", flush=True)
    print(f"F1 (binary, класс 1): {f1_bin:.6f}", flush=True)
    print(f"F1 (macro):           {f1_macro:.6f}", flush=True)
    print(f"F1 (weighted):        {f1_weighted:.6f}", flush=True)
    print(f"ROC-AUC:              {auc:.6f}", flush=True)
    print("\n" + classification_report(y_test, y_pred, digits=4), flush=True)

    if args.save_pipeline:
        args.save_pipeline.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, args.save_pipeline)
        print(f"Пайплайн сохранён: {args.save_pipeline}", flush=True)


if __name__ == "__main__":
    main()
