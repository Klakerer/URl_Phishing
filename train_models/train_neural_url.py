#!/usr/bin/env python3
"""
Обучение бинарного классификатора phishing по **сырой строке URL** (без заранее
извлечённых числовых признаков).

Идея входа: URL → нижний регистр → последовательность индексов символов
(фиксированный алфавит + PAD/UNK, обрезка/дополнение до max_len).

Архитектура: **Embedding → несколько параллельных Conv1d (ядра 3/5/7) →
Global Max Pooling по длине → конкатенация → MLP** — ловит локальные шаблоны
(поддомены, TLD, паттерны пути) сильнее, чем один полносвязный блок по таблице
признаков.

Данные: combined_dataset.csv (колонки url, status), см. train_classical_models.

Запуск из корня репозитория:
  python train_models/train_neural_url.py --max-rows 150000
  python train_models/train_neural_url.py --full --epochs 25 --save models/neural_url_cnn.pt

Из каталога train_models:
  cd train_models && python train_neural_url.py --max-rows 100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# импорты из той же папки train_models
_TRAIN_MODELS_DIR = Path(__file__).resolve().parent
if str(_TRAIN_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_MODELS_DIR))

from train_classical_models import (  # noqa: E402
    read_csv_stratified_subsample,
    read_csv_with_progress,
)

PROJECT_ROOT = _TRAIN_MODELS_DIR.parent
DEFAULT_SAVE_PATH = PROJECT_ROOT / "models" / "neural_url_cnn.pt"

# Алфавит URL (ASCII); остальные символы → UNK
_BASE_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz0123456789"
    "-._~:/?#[]@!$&'()*+,;=%"
)


def build_char_vocab() -> tuple[dict[str, int], int, int, int]:
    """PAD=0, UNK=1, далее символы из _BASE_ALPHABET; возвращает vocab_size."""
    pad_idx, unk_idx = 0, 1
    char2idx: dict[str, int] = {c: i + 2 for i, c in enumerate(_BASE_ALPHABET)}
    vocab_size = len(char2idx) + 2
    return char2idx, pad_idx, unk_idx, vocab_size


def encode_urls(
    urls: list[str],
    char2idx: dict[str, int],
    unk_idx: int,
    max_len: int,
) -> np.ndarray:
    """Матрица (n_urls, max_len) int64, padding справа нулями."""
    out = np.zeros((len(urls), max_len), dtype=np.int64)
    for i, raw in enumerate(urls):
        s = str(raw).lower()[:max_len]
        for j, ch in enumerate(s):
            out[i, j] = char2idx.get(ch, unk_idx)
    return out


class UrlMultiScaleCNN(nn.Module):
    """
    Символьный embedding + multi-scale 1D CNN + GMP + MLP.
    Улучшения относительно табличного MLP: локальные n-граммы по строке URL,
    несколько масштабов свёртки.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_filters: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv_blocks = nn.ModuleList()
        for k in (3, 5, 7):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(embed_dim, n_filters, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(n_filters),
                    nn.ReLU(inplace=True),
                )
            )
        fused = n_filters * len(self.conv_blocks)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) long
        e = self.emb(x).transpose(1, 2)  # (B, E, L)
        pooled: list[torch.Tensor] = []
        for block in self.conv_blocks:
            h = block(e)
            pooled.append(h.amax(dim=2))
        h = torch.cat(pooled, dim=1)
        return self.head(h)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb).squeeze(1)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_logits(
    model: nn.Module,
    X: torch.Tensor,
    _y: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: list[np.ndarray] = []
    for i in range(0, len(X), batch_size):
        chunk = X[i : i + batch_size].to(device)
        logits_list.append(model(chunk).squeeze(1).cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    pred = (proba >= 0.5).astype(np.int64)
    return pred, proba


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN по символам URL (phishing / not).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/danya/datasets/andei_dataset"),
    )
    parser.add_argument(
        "--combined-file",
        type=str,
        default="combined_dataset.csv",
        help="CSV с колонками url, status",
    )
    parser.add_argument("--full", action="store_true", help="Весь combined_dataset")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Стратифицированная подвыборка, если не --full",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=256, help="Макс. длина URL в символах")
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--n-filters", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_SAVE_PATH,
        help="Путь .pt: state_dict + meta (char2idx, max_len, ...)",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    char2idx, _pad_idx, unk_idx, vocab_size = build_char_vocab()

    combined_path = args.data_dir / args.combined_file
    if not combined_path.is_file():
        raise SystemExit(f"Нет файла: {combined_path}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.random_state)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.random_state)

    print("=== URL → multi-scale CNN (символьное представление) ===", flush=True)
    print(f"Данные: {combined_path}", flush=True)
    print(f"Устройство: {device}", flush=True)

    if args.full:
        df = read_csv_with_progress(combined_path, nrows=None)
    else:
        print(f"Подвыборка: {args.max_rows} строк (стратификация по status)", flush=True)
        df = read_csv_stratified_subsample(
            combined_path,
            args.max_rows,
            random_state=args.random_state,
        )

    if "url" not in df.columns or "status" not in df.columns:
        raise SystemExit("Нужны колонки url и status.")
    if df["status"].nunique() < 2:
        raise SystemExit("Нужны оба класса status.")

    urls = df["url"].astype(str).tolist()
    y = df["status"].astype(int).values
    print(f"Строк: {len(df)}", flush=True)

    print("Кодирование URL в индексы символов...", flush=True)
    X_idx = encode_urls(urls, char2idx, unk_idx, args.max_len)

    idx_all = np.arange(len(y))
    idx_train, idx_test, y_train_np, y_test_np = train_test_split(
        idx_all,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    X_train = X_idx[idx_train]
    X_test = X_idx[idx_test]

    rng = np.random.RandomState(args.random_state)
    n_tr = len(X_train)
    n_val = max(int(n_tr * args.val_fraction), 1)
    perm = rng.permutation(n_tr)
    val_local = perm[:n_val]
    train_local = perm[n_val:]

    X_fit = X_train[train_local]
    y_fit = y_train_np[train_local]
    X_val = X_train[val_local]
    y_val = y_train_np[val_local]

    X_fit_t = torch.from_numpy(X_fit)
    y_fit_t = torch.from_numpy(y_fit.astype(np.float32))

    pos = float((y_fit == 1).sum())
    neg = float((y_fit == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    ds_tr = TensorDataset(X_fit_t, y_fit_t)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=min(args.batch_size, len(ds_tr)),
        shuffle=True,
        drop_last=False,
    )

    model = UrlMultiScaleCNN(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        n_filters=args.n_filters,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    bad = 0

    X_val_t = torch.from_numpy(X_val)
    for epoch in tqdm(range(args.epochs), desc="Эпохи"):
        train_one_epoch(model, dl_tr, opt, crit, device)
        y_va_pred, _ = evaluate_logits(
            model, X_val_t, y_val, device, args.batch_size * 2
        )
        f1 = float(f1_score(y_val, y_va_pred, average="binary", zero_division=0))
        if f1 > best_f1 + 1e-6:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                tqdm.write(f"Ранняя остановка на эпохе {epoch + 1} (best val F1={best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    print("\nОдин проход по полному train (лучшие веса + все train-строки)...", flush=True)
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train_np.astype(np.float32))
    ds_full = TensorDataset(X_train_t, y_train_t)
    dl_full = DataLoader(
        ds_full,
        batch_size=min(args.batch_size, len(ds_full)),
        shuffle=True,
        drop_last=False,
    )
    pos_w2 = torch.tensor(
        [float((y_train_np == 0).sum()) / max(float((y_train_np == 1).sum()), 1.0)],
        device=device,
    )
    crit2 = nn.BCEWithLogitsLoss(pos_weight=pos_w2)
    train_one_epoch(model, dl_full, opt, crit2, device)

    y_test_pred, proba_test = evaluate_logits(
        model,
        torch.from_numpy(X_test),
        y_test_np,
        device,
        args.batch_size * 2,
    )

    f1 = float(f1_score(y_test_np, y_test_pred, average="binary"))
    auc = float(roc_auc_score(y_test_np, proba_test))

    print("\n========== Hold-out test ==========", flush=True)
    print(f"F1 (binary): {f1:.6f}", flush=True)
    print(f"ROC-AUC:     {auc:.6f}", flush=True)
    print(classification_report(y_test_np, y_test_pred, digits=4), flush=True)

    if not args.no_save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": model.state_dict(),
            "meta": {
                "vocab_size": vocab_size,
                "embed_dim": args.embed_dim,
                "n_filters": args.n_filters,
                "dropout": args.dropout,
                "max_len": args.max_len,
                "char2idx": char2idx,
                "unk_idx": unk_idx,
                "pad_idx": 0,
                "base_alphabet": _BASE_ALPHABET,
            },
            "metrics": {"test_f1_binary": f1, "test_roc_auc": auc},
        }
        torch.save(payload, args.save)
        print(f"\nЧекпоинт: {args.save}", flush=True)


if __name__ == "__main__":
    main()
