#!/usr/bin/env python3
"""
Подбор гиперпараметров HistGradientBoostingClassifier под целевой столбец status.

Использует тот же CSV и препроцессинг, что и train_classical_models.py
(RandomizedSearchCV + stratified CV по обучающей части, финальная оценка на hold-out).

Примеры:
  python tune_hgb.py --max-rows 150000 --n-iter 25 --cv 3
  python tune_hgb.py --full --n-iter 20 --cv 3 --n-jobs 4
  python tune_hgb.py --full --n-iter 40 --cv 5 --scoring roc_auc \\
      --save-pipeline best_hgb_tuned.joblib --save-results hgb_search.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from train_classical_models import (
    build_preprocessor,
    read_csv_stratified_subsample,
    read_csv_with_progress,
)


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def make_param_distributions() -> dict[str, Any]:
    """Пространство поиска для шага ``model`` в Pipeline."""
    return {
        "model__learning_rate": loguniform(0.01, 0.25),
        "model__max_depth": randint(4, 15),
        "model__max_iter": randint(100, 501),
        "model__min_samples_leaf": randint(5, 151),
        "model__l2_regularization": loguniform(1e-5, 10.0),
        "model__max_bins": [63, 127, 255],
        "model__early_stopping": [True, False],
        "model__validation_fraction": [0.05, 0.1],
        "model__n_iter_no_change": randint(10, 31),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RandomizedSearchCV для HistGradientBoostingClassifier (status).",
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
        "--full",
        action="store_true",
        help="Загрузить весь CSV (долго; --max-rows игнорируется)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Стратифицированная подвыборка из n строк (если не задан --full)",
    )
    parser.add_argument("--cv", type=int, default=3, help="Число фолдов StratifiedKFold")
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Число случайных конфигураций RandomizedSearchCV",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="roc_auc",
        help="Метрика для CV (sklearn scoring name), например roc_auc, f1",
    )
    parser.add_argument(
        "--save-pipeline",
        type=Path,
        default=None,
        help="Сохранить лучший Pipeline (prep + HGB) в .joblib",
    )
    parser.add_argument(
        "--save-results",
        type=Path,
        default=None,
        help="Сохранить лучшие параметры и краткую таблицу в JSON",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Параллельность RandomizedSearchCV (-1 = все ядра)",
    )
    args = parser.parse_args()

    features_path = args.data_dir / args.features_file
    if not features_path.is_file():
        raise SystemExit(f"Нет файла: {features_path}")

    print("=== Подбор гиперпараметров HistGradientBoosting ===", flush=True)
    print(f"Данные: {features_path}", flush=True)

    if args.full:
        print("Режим: полный файл", flush=True)
        df = read_csv_with_progress(features_path, nrows=None)
    else:
        print(
            f"Подвыборка: {args.max_rows} строк (стратификация по status)",
            flush=True,
        )
        df = read_csv_stratified_subsample(
            features_path,
            args.max_rows,
            random_state=args.random_state,
        )

    if "status" not in df.columns:
        raise SystemExit("Нет столбца status.")
    if df["status"].nunique() < 2:
        raise SystemExit("Нужны оба класса status — увеличьте объём данных.")

    y = df["status"].astype(int).values
    X = df.drop(columns=["status"])
    print(f"Строк: {len(df)}, признаков: {X.shape[1]}", flush=True)
    print("Классы:", dict(Counter(y)), flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    base_hgb = HistGradientBoostingClassifier(
        random_state=args.random_state,
    )
    pipe = Pipeline([("prep", preprocessor), ("model", base_hgb)])

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=make_param_distributions(),
        n_iter=args.n_iter,
        scoring=args.scoring,
        cv=args.cv,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        refit=True,
        verbose=2,
        return_train_score=False,
    )

    print(
        f"\nRandomizedSearchCV: n_iter={args.n_iter}, cv={args.cv}, "
        f"scoring={args.scoring!r}",
        flush=True,
    )
    search.fit(X_train, y_train)

    print("\n--- Лучшие параметры (CV) ---", flush=True)
    best_params = _jsonify(dict(search.best_params_))
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}", flush=True)
    print(f"Лучший mean CV {args.scoring}: {search.best_score_:.6f}", flush=True)

    y_pred = search.predict(X_test)
    proba = search.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    print("\n--- Hold-out test ---", flush=True)
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC (test): {auc:.6f}", flush=True)

    if args.save_pipeline:
        args.save_pipeline.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(search.best_estimator_, args.save_pipeline)
        print(f"\nПайплайн сохранён: {args.save_pipeline}", flush=True)

    if args.save_results:
        cv = search.cv_results_
        n_top = min(10, len(cv["mean_test_score"]))
        top_idx = np.argsort(cv["mean_test_score"])[::-1][:n_top]
        rows = []
        for i in top_idx:
            rows.append(
                {
                    "mean_test_score": float(cv["mean_test_score"][i]),
                    "std_test_score": float(cv["std_test_score"][i]),
                    "params": _jsonify(cv["params"][i]),
                }
            )

        out = {
            "best_params": best_params,
            "best_cv_score": float(search.best_score_),
            "test_roc_auc": auc,
            "scoring": args.scoring,
            "n_iter": args.n_iter,
            "cv": args.cv,
            "top_trials": rows,
        }
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        args.save_results.write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Результаты поиска: {args.save_results}", flush=True)


if __name__ == "__main__":
    main()
