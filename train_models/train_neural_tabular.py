#!/usr/bin/env python3
"""
Табличная бинарная классификация status: MLP на PyTorch + препроцессинг как в
train_classical_models.py. RandomizedSearchCV с целевой метрикой F1.

Зависимости:
  pip install torch pandas scikit-learn scipy tqdm

Примеры:
  python train_neural_tabular.py --max-rows 200000 --n-iter 15 --cv 3
  python train_neural_tabular.py --full --n-iter 8 --cv 2 --n-jobs 1 \\
      --save-pipeline nn_tabular.joblib --save-results nn_search.json
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
import torch
import torch.nn as nn
from scipy.special import expit
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from train_classical_models import (
    build_preprocessor,
    read_csv_stratified_subsample,
    read_csv_with_progress,
)


class TabularMLP(nn.Module):
    """Небольшой MLP для плотных табличных признаков (после OHE + scaler)."""

    def __init__(
        self,
        n_features: int,
        hidden_layers: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchTabularMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-совместимый бинарный классификатор (классы 0 и 1).
    Обучение с BCEWithLogitsLoss, ранняя остановка по F1 на валидации.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 2048,
        max_epochs: int = 40,
        patience: int = 5,
        val_fraction: float = 0.1,
        random_state: int = 42,
        device: str | None = None,
        verbose: int = 0,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _device_t(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> TorchTabularMLPClassifier:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64).ravel()
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Ожидаются ровно два класса (0 и 1).")

        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        n = len(X)
        n_val = max(int(n * self.val_fraction), 1)
        n_train = n - n_val
        if n_train < 1:
            raise ValueError("Слишком мало объектов для train/val.")

        idx = rng.permutation(n)
        i_val = idx[:n_val]
        i_tr = idx[n_val:]
        X_tr, y_tr = X[i_tr], y[i_tr]
        X_va, y_va = X[i_val], y[i_val]

        pos = float((y_tr == 1).sum())
        neg = float((y_tr == 0).sum())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=self._device_t())

        ds_tr = TensorDataset(
            torch.from_numpy(X_tr),
            torch.from_numpy(y_tr.astype(np.float32)).unsqueeze(1),
        )
        dl_tr = DataLoader(
            ds_tr,
            batch_size=min(self.batch_size, len(ds_tr)),
            shuffle=True,
            drop_last=False,
        )

        dev = self._device_t()
        self._model = TabularMLP(
            n_features=X.shape[1],
            hidden_layers=tuple(self.hidden_layers),
            dropout=self.dropout,
        ).to(dev)

        opt = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_va_t = torch.from_numpy(X_va).to(dev)
        y_va_t = torch.from_numpy(y_va)

        best_f1 = -1.0
        best_state: dict[str, Any] | None = None
        bad_epochs = 0

        epoch_iter = range(self.max_epochs)
        if self.verbose:
            epoch_iter = tqdm(epoch_iter, desc="MLP epochs", leave=False)

        for _ in epoch_iter:
            self._model.train()
            for xb, yb in dl_tr:
                xb = xb.to(dev)
                yb = yb.to(dev)
                opt.zero_grad(set_to_none=True)
                logits = self._model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()

            self._model.eval()
            with torch.no_grad():
                logits_va = self._model(X_va_t).squeeze(1)
                p_va = torch.sigmoid(logits_va).cpu().numpy()
            pred_va = (p_va >= 0.5).astype(np.int64)
            f1 = float(f1_score(y_va_t.numpy(), pred_va, average="binary", zero_division=0))

            if f1 > best_f1 + 1e-6:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        self.n_features_in_ = X.shape[1]
        return self

    def _forward_logits(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        dev = self._device_t()
        X = np.asarray(X, dtype=np.float32)
        out: list[np.ndarray] = []
        bs = max(self.batch_size, 4096)
        with torch.no_grad():
            for i in range(0, len(X), bs):
                chunk = torch.from_numpy(X[i : i + bs]).to(dev)
                out.append(self._model(chunk).squeeze(1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._forward_logits(X)
        p1 = expit(logits)
        p1 = np.clip(p1, 1e-7, 1.0 - 1e-7)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


def make_param_distributions() -> dict[str, Any]:
    """Компактные архитектуры под ~60 входных признаков после OHE."""
    return {
        "model__hidden_layers": [
            (96, 48),
            (128, 64),
            (128, 64, 32),
            (192, 96, 48),
            (256, 128),
        ],
        "model__dropout": uniform(0.08, 0.32),
        "model__lr": loguniform(3e-4, 5e-3),
        "model__weight_decay": loguniform(1e-6, 5e-3),
        "model__batch_size": [1024, 2048, 4096],
        "model__max_epochs": randint(20, 61),
        "model__patience": randint(4, 12),
        "model__val_fraction": [0.08, 0.1, 0.12],
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch MLP + RandomizedSearchCV (scoring=F1) для status.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/danya/datasets/andei_dataset"),
    )
    parser.add_argument("--features-file", type=str, default="full_features_dataset.csv")
    parser.add_argument("--full", action="store_true", help="Весь CSV (долго)")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=250_000,
        help="Стратифицированная подвыборка, если не задан --full",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Для GPU лучше 1; для CPU можно -1",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu (по умолчанию: cuda если доступен)",
    )
    parser.add_argument("--save-pipeline", type=Path, default=None)
    parser.add_argument("--save-results", type=Path, default=None)
    parser.add_argument(
        "--verbose-epochs",
        action="store_true",
        help="Прогресс-бар эпох внутри каждого fit (шумно при CV)",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)

    features_path = args.data_dir / args.features_file
    if not features_path.is_file():
        raise SystemExit(f"Нет файла: {features_path}")

    print("=== PyTorch MLP (табличные признаки) + подбор по F1 ===", flush=True)
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

    prep = build_preprocessor(X_train)
    base_clf = TorchTabularMLPClassifier(
        random_state=args.random_state,
        device=args.device,
        verbose=1 if args.verbose_epochs else 0,
    )
    pipe = Pipeline([("prep", prep), ("model", base_clf)])

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=make_param_distributions(),
        n_iter=args.n_iter,
        scoring="f1",
        cv=args.cv,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        refit=True,
        verbose=2,
        return_train_score=False,
    )

    print(
        f"\nRandomizedSearchCV: n_iter={args.n_iter}, cv={args.cv}, scoring='f1'",
        flush=True,
    )
    search.fit(X_train, y_train)

    best_params = _jsonify(dict(search.best_params_))
    print("\n--- Лучшие параметры (mean CV F1) ---", flush=True)
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}", flush=True)
    print(f"Лучший mean CV F1: {search.best_score_:.6f}", flush=True)

    y_pred = search.predict(X_test)
    proba = search.predict_proba(X_test)[:, 1]
    f1_bin = float(f1_score(y_test, y_pred, average="binary"))
    auc = float(roc_auc_score(y_test, proba))

    print("\n--- Hold-out test ---", flush=True)
    print(f"F1 (binary): {f1_bin:.6f}", flush=True)
    print(f"ROC-AUC:     {auc:.6f}", flush=True)
    print(classification_report(y_test, y_pred, digits=4), flush=True)

    if args.save_pipeline:
        args.save_pipeline.parent.mkdir(parents=True, exist_ok=True)
        best = search.best_estimator_
        mlp = best.named_steps["model"]
        if getattr(mlp, "_model", None) is not None:
            mlp._model.cpu()
            mlp.device = "cpu"
        joblib.dump(best, args.save_pipeline)
        print(f"\nПайплайн (MLP на CPU): {args.save_pipeline}", flush=True)

    if args.save_results:
        cv = search.cv_results_
        top_idx = np.argsort(cv["mean_test_score"])[::-1][: min(10, len(cv["mean_test_score"]))]
        rows = [
            {
                "mean_test_f1": float(cv["mean_test_score"][i]),
                "std_test_f1": float(cv["std_test_score"][i]),
                "params": _jsonify(cv["params"][i]),
            }
            for i in top_idx
        ]
        out = {
            "best_params": best_params,
            "best_cv_f1": float(search.best_score_),
            "test_f1_binary": f1_bin,
            "test_roc_auc": auc,
            "n_iter": args.n_iter,
            "cv": args.cv,
            "top_trials": rows,
        }
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        args.save_results.write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON: {args.save_results}", flush=True)


if __name__ == "__main__":
    main()
