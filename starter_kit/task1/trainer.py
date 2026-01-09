#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from tqdm import tqdm
from torch.utils.data import Dataset

class VADataset(Dataset):
    """
    PyTorch Dataset for (Aspect, Text) -> (Valence, Arousal) regression.

    Expects a pandas.DataFrame with columns:
        - "Text"
        - "Aspect"
        - (optional) "Valence"
        - (optional) "Arousal"
    """

    def __init__(self, dataframe, tokenizer, max_len: int = 128):
        self.sentences = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()

        self.has_labels = "Valence" in dataframe.columns and "Arousal" in dataframe.columns
        if self.has_labels:
            self.labels = dataframe[["Valence", "Arousal"]].values.astype(float)
        else:
            self.labels = None

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int):
        text = f"{self.aspects[idx]}: {self.sentences[idx]}"
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        accumulate_steps: int = 1,
        max_grad_norm: float | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.accumulate_steps = max(1, int(accumulate_steps))
        self.max_grad_norm = max_grad_norm

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        self.optimizer.zero_grad()
        for step, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()

            # gradient accumulation
            loss = loss / self.accumulate_steps
            loss.backward()

            if step % self.accumulate_steps == 0 or step == len(dataloader):
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

        total_loss = total_loss / len(dataloader)
        return total_loss

    def eval_epoch(self, dataloader) -> Tuple[float, Dict[str, float]]:
        """
        One evaluation epoch.

        Returns:
            (avg_loss, metrics_dict) where metrics_dict contains PCC_V, PCC_A, RMSE_VA.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Eval", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                if "labels" in batch:
                    labels = batch["labels"].to(self.device)
                else:
                    labels = None

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                if labels is not None:
                    loss = outputs.loss
                    total_loss += loss.item()
                    all_labels.append(labels.cpu().numpy())
                    all_preds.append(outputs.logits.cpu().numpy())

        avg_loss = total_loss / max(1, len(dataloader))

        metrics: Dict[str, float] = {}
        if all_preds and all_labels:
            preds = np.vstack(all_preds)
            labels = np.vstack(all_labels)

            pred_v = preds[:, 0]
            pred_a = preds[:, 1]
            gold_v = labels[:, 0]
            gold_a = labels[:, 1]

            metrics = evaluate_predictions_task1(
                pred_a=pred_a,
                pred_v=pred_v,
                gold_a=gold_a,
                gold_v=gold_v,
                is_norm=False,
            )

        return avg_loss, metrics


def predict(model: nn.Module, dataloader, device: torch.device):
    """
    Run prediction on an unlabeled set.

    Returns:
        (pred_v, pred_a) as 1D numpy arrays.
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predict", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_preds.append(outputs.logits.cpu().numpy())

    preds = np.vstack(all_preds)
    pred_v = preds[:, 0]
    pred_a = preds[:, 1]
    return pred_v, pred_a


def evaluate_predictions_task1(
    pred_a, pred_v, gold_a, gold_v, is_norm: bool = False
) -> Dict[str, float]:
    """
    Same metric as the official DimABSA2026 starter kit:
    - Pearson correlation for Valence and Arousal.
    - (Optionally normalized) RMSE over concatenated VA.
    """
    pred_v = np.asarray(pred_v, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    gold_v = np.asarray(gold_v, dtype=float)
    gold_a = np.asarray(gold_a, dtype=float)

    if not (np.all((1 <= pred_v) & (pred_v <= 9)) and np.all((1 <= pred_a) & (pred_a <= 9))):
        print("Warning: Some predicted values are out of the [1, 9] numerical range.")

    pcc_v = pearsonr(pred_v, gold_v)[0]
    pcc_a = pearsonr(pred_a, gold_a)[0]

    gold_va = np.concatenate([gold_v, gold_a])
    pred_va = np.concatenate([pred_v, pred_a])

    diff_sq = (gold_va - pred_va) ** 2
    mse_va = float(np.mean(diff_sq))

    if is_norm:
        rmse_va = math.sqrt(mse_va) / math.sqrt(128.0)
    else:
        rmse_va = math.sqrt(mse_va)

    return {
        "PCC_V": float(pcc_v),
        "PCC_A": float(pcc_a),
        "RMSE_VA": float(rmse_va),
    }
