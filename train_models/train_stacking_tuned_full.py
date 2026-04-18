#!/usr/bin/env python3
"""
Финальное обучение StackingClassifier на полном CSV с теми же базовыми
моделями и гиперпараметрами, что после успешного прогона
train_stacking_ensemble.py (см. TUNED_BASE_PARAMS + STACK_META).

Стратифицированный train/test: стэк обучается на train, метрики на test.

Пример:
  python train_stacking_tuned_full.py
  python train_stacking_tuned_full.py --test-size 0.15 --stack-cv 5 \\
      --save-stack stack_full.joblib
  python train_stacking_tuned_full.py --report-json stacking_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib

from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from train_classical_models import build_preprocessor, read_csv_with_progress

# Лучшие параметры баз из лога train_stacking_ensemble.py (200k подвыборка)
TUNED_BASE_PARAMS: dict[str, dict[str, Any]] = {
    "logistic": {"C": 7.99752794786751},
    "random_forest": {
        "n_estimators": 285,
        "max_depth": 42,
        "min_samples_leaf": 3,
        "max_features": 0.6,
    },
    "hist_gbrt": {
        "learning_rate": 0.11439465484333894,
        "max_depth": 11,
        "max_iter": 334,
        "min_samples_leaf": 66,
        "l2_regularization": 1.6130950043034935,
        "max_bins": 255,
    },
    "sgd_logit": {"alpha": 3.775887545682687e-07},
}

STACK_CV_DEFAULT = 5


def _strip_model_prefix(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = k.removeprefix("model__") if k.startswith("model__") else k
        out[key] = v
    return out


def load_base_params_from_report(path: Path) -> dict[str, dict[str, Any]]:
    """Читает stacking_report.json (--save-report из train_stacking_ensemble.py)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models")
    if not isinstance(models, dict):
        raise ValueError("В JSON нет секции models")
    out: dict[str, dict[str, Any]] = {}
    for name, block in models.items():
        if not isinstance(block, dict):
            continue
        bp = block.get("best_params")
        if isinstance(bp, dict):
            out[name] = _strip_model_prefix(bp)
    if len(out) < 2:
        raise ValueError("В models недостаточно best_params")
    return out


def build_estimators(
    prep_template: Any,
    random_state: int,
    base_params: dict[str, dict[str, Any]],
    rf_n_jobs: int,
) -> list[tuple[str, Pipeline]]:
    """Собирает четыре Pipeline(prep, model) с заданными гиперпараметрами."""
    p = {k: deepcopy(v) for k, v in base_params.items()}

    estimators: list[tuple[str, Pipeline]] = [
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
                            C=float(p["logistic"]["C"]),
                        ),
                    ),
                ]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                [
                    ("prep", clone(prep_template)),
                    (
                        "model",
                        RandomForestClassifier(
                            random_state=random_state,
                            n_jobs=rf_n_jobs,
                            class_weight="balanced_subsample",
                            n_estimators=int(p["random_forest"]["n_estimators"]),
                            max_depth=int(p["random_forest"]["max_depth"]),
                            min_samples_leaf=int(p["random_forest"]["min_samples_leaf"]),
                            max_features=p["random_forest"]["max_features"],
                        ),
                    ),
                ]
            ),
        ),
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
                            learning_rate=float(p["hist_gbrt"]["learning_rate"]),
                            max_depth=int(p["hist_gbrt"]["max_depth"]),
                            max_iter=int(p["hist_gbrt"]["max_iter"]),
                            min_samples_leaf=int(p["hist_gbrt"]["min_samples_leaf"]),
                            l2_regularization=float(p["hist_gbrt"]["l2_regularization"]),
                            max_bins=int(p["hist_gbrt"]["max_bins"]),
                        ),
                    ),
                ]
            ),
        ),
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
                            alpha=float(p["sgd_logit"]["alpha"]),
                        ),
                    ),
                ]
            ),
        ),
    ]
    return estimators


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Стэк с зафиксированными гиперпараметрами баз: полный CSV.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/danya/datasets/andei_dataset"),
    )
    parser.add_argument("--features-file", type=str, default="full_features_dataset.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--stack-cv",
        type=int,
        default=STACK_CV_DEFAULT,
        help="CV у StackingClassifier",
    )
    parser.add_argument(
        "--n-jobs-stack",
        type=int,
        default=-1,
        help="n_jobs у StackingClassifier",
    )
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=-1,
        help="n_jobs у RandomForest внутри базы",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="stacking_report.json: подставить best_params баз вместо встроенных",
    )
    parser.add_argument("--save-stack", type=Path, default=None)
    args = parser.parse_args()

    features_path = args.data_dir / args.features_file
    if not features_path.is_file():
        raise SystemExit(f"Нет файла: {features_path}")

    base_params = deepcopy(TUNED_BASE_PARAMS)
    if args.report_json is not None:
        if not args.report_json.is_file():
            raise SystemExit(f"Нет файла: {args.report_json}")
        loaded = load_base_params_from_report(args.report_json)
        for name, kw in loaded.items():
            if name in base_params:
                base_params[name].update(kw)
        print(f"Параметры баз дополнены из {args.report_json}", flush=True)

    print("=== StackingClassifier: полный датасет, зафиксированные базы ===", flush=True)
    print(f"CSV: {features_path}", flush=True)

    df = read_csv_with_progress(features_path, nrows=None)
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
        f"Train: {len(X_train)} | Test: {len(X_test)} | "
        f"test_size={args.test_size} | stack_cv={args.stack_cv}",
        flush=True,
    )

    prep_template = build_preprocessor(X_train)
    estimators = build_estimators(
        prep_template,
        args.random_state,
        base_params,
        rf_n_jobs=args.rf_n_jobs,
    )

    meta = LogisticRegression(
        max_iter=3000,
        random_state=args.random_state,
        class_weight="balanced",
        solver="lbfgs",
    )
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=args.stack_cv,
        stack_method="predict_proba",
        n_jobs=args.n_jobs_stack,
        passthrough=False,
    )

    print("\nОбучение стэка (долго на полном объёме)...", flush=True)
    stack.fit(X_train, y_train)

    y_pred = stack.predict(X_test)
    proba = stack.predict_proba(X_test)[:, 1]
    f1_bin = float(f1_score(y_test, y_pred, average="binary"))
    auc = float(roc_auc_score(y_test, proba))

    print("\n========== Test ==========", flush=True)
    print(f"F1 (binary): {f1_bin:.6f}", flush=True)
    print(f"ROC-AUC:     {auc:.6f}", flush=True)
    print(classification_report(y_test, y_pred, digits=4), flush=True)

    if args.save_stack:
        args.save_stack.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(stack, args.save_stack)
        print(f"\nСтэк сохранён: {args.save_stack}", flush=True)


if __name__ == "__main__":
    main()
