#!/usr/bin/env python3
"""
Стэкинг нескольких классических моделей для предсказания status.

Для каждой базовой модели: Pipeline(prep + модель) и RandomizedSearchCV
(scoring=F1). Затем StackingClassifier с LogisticRegression как
final_estimator. Всё в одном файле.

Зависимости:
  pip install pandas scikit-learn scipy tqdm

Примеры:
  python train_stacking_ensemble.py --max-rows 200000 --n-iter-per-model 12
  python train_stacking_ensemble.py --full --n-iter-per-model 8 --stack-cv 5 \\
      --save-stack stack_model.joblib --save-report stacking_report.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
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
    if value is None:
        return None
    return value


def make_tuning_jobs(
    prep_template: ColumnTransformer,
    random_state: int,
) -> list[tuple[str, Pipeline, dict[str, Any]]]:
    """
    Список (имя, pipeline-шаблон, param_distributions для RandomizedSearchCV).
    У всех шаг модели называется ``model`` (префикс model__ в сетке).
    """
    # n_jobs=1 у древесных моделей, чтобы не раздувать вложенный параллелизм при cv>1
    jobs: list[tuple[str, Pipeline, dict[str, Any]]] = []

    jobs.append(
        (
            "logistic",
            Pipeline(
                [
                    ("prep", clone(prep_template)),
                    (
                        "model",
                        LogisticRegression(
                            solver="saga",
                            max_iter=2500,
                            random_state=random_state,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            {
                "model__C": loguniform(0.03, 80.0),
            },
        )
    )

    jobs.append(
        (
            "random_forest",
            Pipeline(
                [
                    ("prep", clone(prep_template)),
                    (
                        "model",
                        RandomForestClassifier(
                            random_state=random_state,
                            n_jobs=1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": randint(80, 301),
                "model__max_depth": [None, 18, 26, 34, 42],
                "model__min_samples_leaf": randint(1, 12),
                "model__max_features": ["sqrt", 0.4, 0.6],
            },
        )
    )

    jobs.append(
        (
            "hist_gbrt",
            Pipeline(
                [
                    ("prep", clone(prep_template)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            random_state=random_state,
                            early_stopping=True,
                            validation_fraction=0.08,
                            n_iter_no_change=15,
                        ),
                    ),
                ]
            ),
            {
                "model__learning_rate": loguniform(0.02, 0.22),
                "model__max_depth": randint(4, 14),
                "model__max_iter": randint(120, 400),
                "model__min_samples_leaf": randint(5, 80),
                "model__l2_regularization": loguniform(1e-6, 5.0),
                "model__max_bins": [63, 127, 255],
            },
        )
    )

    jobs.append(
        (
            "sgd_logit",
            Pipeline(
                [
                    ("prep", clone(prep_template)),
                    (
                        "model",
                        SGDClassifier(
                            loss="log_loss",
                            random_state=random_state,
                            max_iter=2500,
                            tol=1e-3,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_jobs=1,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            {
                "model__alpha": loguniform(1e-7, 5e-4),
            },
        )
    )

    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Подбор гиперпараметров базовых моделей + StackingClassifier (F1).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/danya/datasets/andei_dataset"),
    )
    parser.add_argument("--features-file", type=str, default="full_features_dataset.csv")
    parser.add_argument("--full", action="store_true", help="Весь CSV")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Стратифицированная подвыборка, если не --full",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-iter-per-model",
        type=int,
        default=15,
        help="Итераций RandomizedSearchCV на каждую базовую модель",
    )
    parser.add_argument("--cv-tune", type=int, default=3, help="CV при подборе баз")
    parser.add_argument(
        "--stack-cv",
        type=int,
        default=5,
        help="CV у StackingClassifier для out-of-fold признаков мета-модели",
    )
    parser.add_argument(
        "--n-jobs-search",
        type=int,
        default=1,
        help="n_jobs у RandomizedSearchCV (1 надёжнее при больших данных)",
    )
    parser.add_argument(
        "--n-jobs-stack",
        type=int,
        default=-1,
        help="n_jobs у StackingClassifier",
    )
    parser.add_argument("--save-stack", type=Path, default=None)
    parser.add_argument("--save-report", type=Path, default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    features_path = args.data_dir / args.features_file
    if not features_path.is_file():
        raise SystemExit(f"Нет файла: {features_path}")

    print("=== Стэкинг: подбор баз + StackingClassifier ===", flush=True)
    print(f"Данные: {features_path}", flush=True)

    if args.full:
        df = read_csv_with_progress(features_path, nrows=None)
    else:
        print(f"Подвыборка: {args.max_rows} строк (стратификация)", flush=True)
        df = read_csv_stratified_subsample(
            features_path,
            args.max_rows,
            random_state=args.random_state,
        )

    if "status" not in df.columns or df["status"].nunique() < 2:
        raise SystemExit("Нужен столбец status с двумя классами.")

    y = df["status"].astype(int).values
    X = df.drop(columns=["status"])
    print(f"Строк: {len(df)}, признаков (сырых): {X.shape[1]}", flush=True)
    print("Классы:", dict(Counter(y)), flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    print(
        f"Train: {len(X_train)} | Test: {len(X_test)} | test_size={args.test_size}",
        flush=True,
    )

    prep_template = build_preprocessor(X_train)
    tuning_jobs = make_tuning_jobs(prep_template, args.random_state)

    tuning_report: dict[str, Any] = {"models": {}}
    fitted_bases: list[tuple[str, Any]] = []

    print("\n--- Этап 1: RandomizedSearchCV для каждой базовой модели (scoring=F1) ---", flush=True)
    for name, pipe_template, param_dist in tuning_jobs:
        print(f"\n>>> Подбор: {name}", flush=True)
        search = RandomizedSearchCV(
            estimator=pipe_template,
            param_distributions=param_dist,
            n_iter=args.n_iter_per_model,
            scoring="f1",
            cv=args.cv_tune,
            n_jobs=args.n_jobs_search,
            random_state=args.random_state,
            refit=True,
            verbose=1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        print(f"    Лучший CV F1: {search.best_score_:.6f}", flush=True)
        print(f"    Лучшие params: {_jsonify(search.best_params_)}", flush=True)
        tuning_report["models"][name] = {
            "best_cv_f1": float(search.best_score_),
            "best_params": _jsonify(search.best_params_),
        }
        fitted_bases.append((name, search.best_estimator_))

    print("\n--- Этап 2: StackingClassifier (мета: LogisticRegression) ---", flush=True)
    meta = LogisticRegression(
        max_iter=3000,
        random_state=args.random_state,
        class_weight="balanced",
        solver="lbfgs",
    )
    stack = StackingClassifier(
        estimators=fitted_bases,
        final_estimator=meta,
        cv=args.stack_cv,
        stack_method="predict_proba",
        n_jobs=args.n_jobs_stack,
        passthrough=False,
    )
    stack.fit(X_train, y_train)

    y_pred = stack.predict(X_test)
    proba = stack.predict_proba(X_test)[:, 1]
    f1 = float(f1_score(y_test, y_pred, average="binary"))
    auc = float(roc_auc_score(y_test, proba))

    print("\n========== Hold-out test (стэк) ==========", flush=True)
    print(f"F1 (binary): {f1:.6f}", flush=True)
    print(f"ROC-AUC:     {auc:.6f}", flush=True)
    print(classification_report(y_test, y_pred, digits=4), flush=True)

    tuning_report["stack"] = {
        "stack_cv": args.stack_cv,
        "test_f1_binary": f1,
        "test_roc_auc": auc,
    }

    if args.save_stack:
        args.save_stack.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(stack, args.save_stack)
        print(f"\nСтэк сохранён: {args.save_stack}", flush=True)

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(
            json.dumps(tuning_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Отчёт: {args.save_report}", flush=True)


if __name__ == "__main__":
    main()
