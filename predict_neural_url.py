#!/usr/bin/env python3
"""
Инференс модели из train_models/train_neural_url.py (символьный CNN по строке URL).

Вход: одна строка URL в терминале (или --url). Выход: класс 0/1 и вероятность фишинга.

Примеры:
  python predict_neural_url.py
  python predict_neural_url.py --url "https://example.com/path"
  python predict_neural_url.py --model models/neural_url_cnn.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
_TRAIN_MODELS_DIR = SCRIPT_DIR / "train_models"
if str(_TRAIN_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_MODELS_DIR))

from train_neural_url import UrlMultiScaleCNN, encode_urls  # noqa: E402

DEFAULT_MODEL_PATH = SCRIPT_DIR / "models" / "neural_url_cnn.pt"


def load_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    if not ckpt_path.is_file():
        raise SystemExit(f"Файл модели не найден: {ckpt_path}")
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location=device)
    if not isinstance(payload, dict) or "model_state" not in payload or "meta" not in payload:
        raise SystemExit(
            "Неверный формат чекпоинта: ожидаются ключи model_state и meta "
            "(сохранение из train_neural_url.py)."
        )
    meta = payload["meta"]
    required = ("vocab_size", "embed_dim", "n_filters", "dropout", "max_len", "char2idx", "unk_idx")
    missing = [k for k in required if k not in meta]
    if missing:
        raise SystemExit(f"В meta не хватает полей: {missing}")

    model = UrlMultiScaleCNN(
        vocab_size=int(meta["vocab_size"]),
        embed_dim=int(meta["embed_dim"]),
        n_filters=int(meta["n_filters"]),
        dropout=float(meta["dropout"]),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, meta


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    url: str,
    meta: dict,
    device: torch.device,
) -> tuple[int, float]:
    char2idx = meta["char2idx"]
    unk_idx = int(meta["unk_idx"])
    max_len = int(meta["max_len"])
    x = encode_urls([url], char2idx, unk_idx, max_len)
    xb = torch.from_numpy(x).to(device)
    logit = model(xb).squeeze(1)
    logit_f = float(logit.item())
    logit_clamped = max(-50.0, min(50.0, logit_f))
    proba = 1.0 / (1.0 + np.exp(-logit_clamped))
    pred = 1 if proba >= 0.5 else 0
    return pred, float(proba)


def print_result(url: str, pred: int, proba: float) -> None:
    label = "фишинг (1)" if pred == 1 else "не фишинг (0)"
    print(f"URL: {url}")
    print(f"Класс: {pred} — {label}")
    print(f"P(фишинг): {proba:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Предсказание phishing по URL (neural_url_cnn).")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Путь к .pt от train_neural_url.py",
    )
    parser.add_argument("--url", type=str, default=None, help="Один URL; иначе интерактивный ввод")
    parser.add_argument("--device", type=str, default=None, help="cpu или cuda (по умолчанию авто)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, meta = load_model(args.model, device)

    if args.url is not None:
        u = args.url.strip()
        if not u:
            raise SystemExit("Пустой --url")
        pred, proba = predict_one(model, u, meta, device)
        print_result(u, pred, proba)
        return

    print("Модель загружена:", args.model.resolve())
    print("Устройство:", device)
    print("Введите URL для проверки (пустая строка — выход).\n")

    while True:
        try:
            line = input("URL> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        pred, proba = predict_one(model, line, meta, device)
        print_result(line, pred, proba)
        print()


if __name__ == "__main__":
    main()
