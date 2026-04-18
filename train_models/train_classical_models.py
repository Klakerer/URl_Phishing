#!/usr/bin/env python3
"""
Обучение нескольких классических ML-моделей для предсказания столбца `status`
по признакам из full_features_dataset.csv (строки совпадают с combined_dataset).

Запуск:
  pip install pandas scikit-learn tqdm
  python train_classical_models.py
  python train_classical_models.py --max-rows 50000   # подвыборка со стратификацией по status
  python train_classical_models.py --save-best best_pipeline.joblib --metrics-json metrics.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from tqdm.auto import tqdm

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in feature_frame.columns if c != "tld"]
    categorical_cols = ["tld"] if "tld" in feature_frame.columns else []

    transformers: list[tuple[str, Pipeline, list[str]]] = [
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_cols,
        )
    ]
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "ohe",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                max_categories=40,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers)


def read_csv_with_progress(
    path: Path,
    nrows: int | None,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Читает CSV по чанкам, чтобы tqdm показывал прогресс на больших файлах."""
    kwargs: dict[str, Any] = {"chunksize": chunk_size}
    if nrows is not None:
        kwargs["nrows"] = nrows
    reader = pd.read_csv(path, **kwargs)

    chunks: list[pd.DataFrame] = []
    desc = f"Чтение CSV ({path.name})"
    if nrows is not None:
        desc += f", до {nrows} строк"
    for chunk in tqdm(reader, desc=desc, unit="chunk"):
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    if len(chunks) == 1:
        return chunks[0]
    return pd.concat(chunks, ignore_index=True)


def read_csv_stratified_subsample(
    path: Path,
    n: int,
    *,
    chunk_size: int = 100_000,
    random_state: int,
) -> pd.DataFrame:
    """
    Берёт n строк с сохранением обоих классов status: читает файл по чанкам,
    пока не встретятся оба класса и не накопится хотя бы n строк, затем
    train_test_split(..., stratify=status, train_size=n).
    """
    if n < 2:
        raise SystemExit("--max-rows должен быть не меньше 2.")

    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(path, chunksize=chunk_size)
    desc = f"Чтение CSV ({path.name}) → подвыборка n={n}"
    for chunk in tqdm(reader, desc=desc, unit="chunk"):
        chunks.append(chunk)
        acc = pd.concat(chunks, ignore_index=True)
        if acc["status"].nunique() >= 2 and len(acc) >= n:
            break

    df_all = pd.concat(chunks, ignore_index=True)
    if df_all["status"].nunique() < 2:
        raise SystemExit(
            "В прочитанных из CSV данных только один класс `status`. "
            "Первые чанки файла однородны — увеличьте --max-rows или уберите его."
        )
    if len(df_all) <= n:
        return df_all

    df_sub, _ = train_test_split(
        df_all,
        train_size=n,
        stratify=df_all["status"],
        random_state=random_state,
    )
    return df_sub.reset_index(drop=True)


def make_models(random_state: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            solver="saga",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=random_state,
            learning_rate=0.08,
            max_depth=8,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.05,
            n_iter_no_change=15,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=20,
            random_state=random_state,
            class_weight="balanced",
        ),
        "sgd_log_loss": SGDClassifier(
            loss="log_loss",
            random_state=random_state,
            max_iter=2000,
            tol=1e-3,
            n_jobs=-1,
            early_stopping=True,
            validation_fraction=0.05,
        ),
        "gaussian_nb": GaussianNB(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обучение и сравнение классических моделей (status)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/danya/datasets/andei_dataset"),
        help="Каталог с CSV",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default="full_features_dataset.csv",
        help="Имя файла с признаками (по умолчанию самый полный)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100000,
        help=(
            "Ограничить объём данных: стратифицированная выборка из n строк "
            "(чтение по чанкам, пока не появятся оба класса status)"
        ),
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Сохранить сводку метрик в JSON",
    )
    parser.add_argument(
        "--save-best",
        type=Path,
        default=None,
        help="Сохранить лучший Pipeline (prep + модель) в .joblib",
    )
    args = parser.parse_args()

    features_path = args.data_dir / args.features_file
    if not features_path.is_file():
        raise SystemExit(f"Нет файла: {features_path}")

    print("=== URL Phishing: обучение классических моделей ===", flush=True)
    print(f"Файл признаков: {features_path}", flush=True)
    if args.max_rows is not None:
        print(
            f"Режим подвыборки: {args.max_rows} строк (стратификация по status)",
            flush=True,
        )

    if args.max_rows is None:
        df = read_csv_with_progress(features_path, nrows=None)
    else:
        df = read_csv_stratified_subsample(
            features_path,
            args.max_rows,
            random_state=args.random_state,
        )
    if "status" not in df.columns:
        raise SystemExit("В датасете нет столбца 'status'.")

    n_rows, n_cols = df.shape
    print(f"Загружено строк: {n_rows}, столбцов (включая target): {n_cols}", flush=True)
    status_counts = df["status"].value_counts().sort_index()
    print("Распределение status:", status_counts.to_dict(), flush=True)

    y = df["status"].astype(int).values
    X = df.drop(columns=["status"])
    n_features = X.shape[1]
    print(f"Признаков для обучения: {n_features}", flush=True)

    print(
        f"Разбиение train/test: test_size={args.test_size}, "
        f"random_state={args.random_state}",
        flush=True,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    print(
        f"Train: {len(X_train)} | Test: {len(X_test)} | "
        f"классы train: {dict(Counter(y_train))}",
        flush=True,
    )

    print("Сборка препроцессора (числовые + tld)...", flush=True)
    preprocessor_template = build_preprocessor(X_train)
    models = make_models(args.random_state)
    print(f"Моделей к обучению: {len(models)}", flush=True)

    results: list[dict[str, Any]] = []

    model_iter = tqdm(
        models.items(),
        desc="Модели",
        unit="model",
        total=len(models),
    )
    for name, estimator in model_iter:
        model_iter.set_postfix_str(name[:24], refresh=False)
        tqdm.write(f"\n>>> Модель: {name} — fit на train...")
        pipe = Pipeline(
            steps=[
                ("prep", clone(preprocessor_template)),
                ("model", clone(estimator)),
            ],
            memory=None,
        )
        pipe.fit(X_train, y_train)
        tqdm.write(f"    {name}: predict на test...")
        y_pred = pipe.predict(X_test)
        proba = None
        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_test)[:, 1]
            except (IndexError, AttributeError):
                proba = None

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="binary"))
        metrics = {"accuracy": acc, "f1_binary": f1}
        if proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            except ValueError:
                metrics["roc_auc"] = None

        tqdm.write(classification_report(y_test, y_pred, digits=4))
        tqdm.write(json.dumps(metrics, ensure_ascii=False, indent=2))
        short = f"acc={acc:.4f} f1={f1:.4f}"
        if metrics.get("roc_auc") is not None:
            short += f" auc={metrics['roc_auc']:.4f}"
        model_iter.set_postfix_str(short[:35], refresh=True)

        results.append({"name": name, "metrics": metrics, "pipeline": pipe})

    def score_key(row: dict[str, Any]) -> float:
        m = row["metrics"]
        auc = m.get("roc_auc")
        if auc is not None:
            return float(auc)
        return float(m["f1_binary"])

    best = max(results, key=score_key)
    print("\n========== Лучшая модель (по ROC-AUC, иначе по F1) ==========", flush=True)
    print(best["name"], flush=True)
    print(json.dumps(best["metrics"], ensure_ascii=False, indent=2), flush=True)

    summary = {
        "best_model": best["name"],
        "best_metrics": best["metrics"],
        "all": [
            {"name": r["name"], "metrics": r["metrics"]} for r in results
        ],
        "features_file": str(features_path),
        "n_rows": int(len(df)),
        "test_size": args.test_size,
    }
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nМетрики сохранены: {args.metrics_json}", flush=True)

    if args.save_best:
        print(f"Сохранение лучшего пайплайна в {args.save_best}...", flush=True)
        args.save_best.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best["pipeline"], args.save_best)
        print(f"Лучшая модель сохранена: {args.save_best}", flush=True)

    print("\nГотово.", flush=True)


if __name__ == "__main__":
    main()
