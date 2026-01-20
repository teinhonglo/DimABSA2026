#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import model as model_module
from trainer import Trainer, VADataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train VA regressor for DimABSA2026")
    parser.add_argument("--train_json", required=True, help="Path to train.json from data_prep.py")
    parser.add_argument("--dev_json", required=True, help="Path to dev.json from data_prep.py")
    parser.add_argument("--model_conf", required=True, help="JSON file containing trainer_args and model_args")
    parser.add_argument("--exp_dir", required=True, help="Directory to store model checkpoints and logs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint to load model weights from",
    )
    args = parser.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)

    # Load configuration
    with open(args.model_conf, "r", encoding="utf-8") as f:
        conf = json.load(f)

    trainer_args = conf.get("trainer_args", {})
    model_args = conf.get("model_args", {})
    model_class_name = model_args.get("model_class", "TransformerVARegressor")

    seed = int(trainer_args.get("seed", 42))
    set_seed(seed)

    # If there is metadata next to the train_json, copy it into exp_dir for later stages.
    data_dir = os.path.dirname(os.path.abspath(args.train_json))
    meta_src = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_src):
        meta_dst = os.path.join(args.exp_dir, "meta.json")
        shutil.copy(meta_src, meta_dst)
        print(f"[train] Copied meta.json to {meta_dst}")

    # Load data
    with open(args.train_json, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(args.dev_json, "r", encoding="utf-8") as f:
        dev_records = json.load(f)

    train_df = pd.DataFrame(train_records)
    dev_df = pd.DataFrame(dev_records)

    model_name = model_args.get("model_name", "bert-base-multilingual-cased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_batch_size = int(trainer_args.get("train_batch_size", 64))
    eval_batch_size = int(trainer_args.get("eval_batch_size", 64))
    max_len = int(trainer_args.get("max_len", 128))
    num_epochs = int(trainer_args.get("num_epochs", 5))
    lr = float(trainer_args.get("learning_rate", 1e-5))
    accumulate_steps = int(trainer_args.get("accumulate_steps", 1))

    train_dataset = VADataset(train_df, tokenizer, max_len=max_len)
    dev_dataset = VADataset(dev_df, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    try:
        ModelClass = getattr(model_module, model_class_name)
    except AttributeError:
        raise ValueError(
            f"[train] Unknown model_class='{model_class_name}' in model_args. "
            f"Please check your config or model.py."
        )

    model = ModelClass(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if args.checkpoint is not None:
        print(f"[train.py] Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)

        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"[train.py] checkpoint loaded with strict=False")
        if missing_keys:
            print(f"[train.py] Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"[train.py] Unexpected keys: {unexpected_keys}")

    trainer = Trainer(
                model=model, 
                optimizer=optimizer,
                device=device, 
                accumulate_steps=accumulate_steps
            )

    best_rmse = float("inf")
    best_state_dict = None
    history = []

    for epoch in range(1, num_epochs + 1):
        print(f"[train] Epoch {epoch}/{num_epochs}")
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_metrics = trainer.eval_epoch(dev_loader)
        log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        }
        log.update({k: float(v) for k, v in val_metrics.items()})
        history.append(log)

        rmse = val_metrics.get("RMSE_VA", float("inf"))
        print(f"[train] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, metrics={val_metrics}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"[train] New best RMSE_VA={best_rmse:.4f}")

    if best_state_dict is None:
        best_state_dict = model.state_dict()

    # Save best model weights
    model_path = os.path.join(args.exp_dir, "best_model.pt")
    torch.save(best_state_dict, model_path)
    print(f"[train] Saved best model to {model_path}")

    # Save tokenizer
    tokenizer.save_pretrained(args.exp_dir)
    print(f"[train] Saved tokenizer to {args.exp_dir}")

    # Save configs and training history
    with open(os.path.join(args.exp_dir, "model_args.json"), "w", encoding="utf-8") as f:
        json.dump(model_args, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.exp_dir, "trainer_args.json"), "w", encoding="utf-8") as f:
        json.dump(trainer_args, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.exp_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[train] Training history saved to training_log.json")


if __name__ == "__main__":
    main()
