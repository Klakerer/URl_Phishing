#!/usr/bin/env python3
"""
Финальное обучение PyTorch MLP (как в train_neural_tabular.py) на полном CSV
с гиперпараметрами из лучшего прогона RandomizedSearchCV (см. TUNED_MLP_KWARGS).

Стратифицированный train/test: обучение на train, F1 и отчёт на test.

Пример:
  python train_neural_tuned_full.py
  python train_neural_tuned_full.py --test-size 0.15
  python train_neural_tuned_full.py --no-save
  python train_neural_tuned_full.py --save-pipeline models/custom.joblib
  python train_neural_tuned_full.py --params-json nn_search.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "models" / "neural_tabular_mlp.joblib"

from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from train_classical_models import build_preprocessor, read_csv_with_progress
from train_neural_tabular import TorchTabularMLPClassifier

# Лучший кандидат из train_neural_tabular.py (лог терминала)
TUNED_MLP_KWARGS: dict[str, Any] = {
    "hidden_layers": (192, 96, 48),
    "dropout": 0.10081650975528944,
    "lr": 0.004249612118210752,
    "weight_decay": 1.3389279116695482e-05,
    "batch_size": 1024,
    "max_epochs": 33,
    "patience": 5,
    "val_fraction": 0.1,
}


def load_mlp_kwargs_from_json(path: Path) -> dict[str, Any]:
    """best_params из train_neural_tabular.py --save-results."""
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("best_params") or data
    if not isinstance(raw, dict):
        raise ValueError("Ожидается JSON с полем best_params или словарём параметров.")
    out: dict[str, Any] = {}
    for k, v in raw.items():
        key = k.removeprefix("model__") if k.startswith("model__") else k
        if key == "hidden_layers" and isinstance(v, list):
            v = tuple(int(x) for x in v)
        if key in ("batch_size", "max_epochs", "patience"):
            v = int(v)
        out[key] = v
    allowed = set(TUNED_MLP_KWARGS) | {"random_state", "device", "verbose"}
    out = {k: v for k, v in out.items() if k in allowed}
    if "hidden_layers" not in out:
        raise ValueError("В JSON нет hidden_layers / model__hidden_layers.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLP с подобранными гиперпараметрами: полный датасет, F1 на test.",
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
        "--params-json",
        type=Path,
        default=None,
        help="JSON от train_neural_tabular.py --save-results",
    )
    parser.add_argument(
        "--save-pipeline",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Куда сохранить пайплайн (по умолчанию: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять модель на диск",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda или cpu (по умолчанию авто)",
    )
    parser.add_argument(
        "--verbose-epochs",
        action="store_true",
        help="Прогресс-бар эпох внутри fit",
    )
    args = parser.parse_args()

    features_path = args.data_dir / args.features_file
    if not features_path.is_file():
        raise SystemExit(f"Нет файла: {features_path}")

    if args.params_json is not None:
        if not args.params_json.is_file():
            raise SystemExit(f"Нет файла: {args.params_json}")
        mlp_kw = load_mlp_kwargs_from_json(args.params_json)
        print(f"Параметры MLP из {args.params_json}", flush=True)
    else:
        mlp_kw = dict(TUNED_MLP_KWARGS)

    mlp_kw["random_state"] = args.random_state
    if args.device is not None:
        mlp_kw["device"] = args.device
    mlp_kw["verbose"] = 1 if args.verbose_epochs else 0

    print("=== PyTorch MLP: полный датасет, подобранные гиперпараметры ===", flush=True)
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
        f"Train: {len(X_train)} | Test: {len(X_test)} | test_size={args.test_size}",
        flush=True,
    )

    prep = build_preprocessor(X_train)
    clf = TorchTabularMLPClassifier(**mlp_kw)
    pipe = Pipeline([("prep", prep), ("model", clf)])

    print("\nОбучение (fit на train, может занять много времени)...", flush=True)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    f1_bin = float(f1_score(y_test, y_pred, average="binary"))
    auc = float(roc_auc_score(y_test, proba))

    print("\n========== Test (отложенная выборка) ==========", flush=True)
    print(f"F1 (binary): {f1_bin:.6f}", flush=True)
    print(f"ROC-AUC:     {auc:.6f}", flush=True)
    print(classification_report(y_test, y_pred, digits=4), flush=True)

    if not args.no_save:
        args.save_pipeline.parent.mkdir(parents=True, exist_ok=True)
        best = pipe
        mlp = best.named_steps["model"]
        if getattr(mlp, "_model", None) is not None:
            mlp._model.cpu()
            mlp.device = "cpu"
        joblib.dump(best, args.save_pipeline)
        print(f"Пайплайн (MLP на CPU): {args.save_pipeline}", flush=True)


if __name__ == "__main__":
    main()
