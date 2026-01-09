#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def evaluate_predictions_task1(
    pred_a, pred_v, gold_a, gold_v, is_norm: bool = False
):
    """
    計算 DimABSA task1 的評估指標:
      - PCC_V: Valence 的 Pearson 相關係數
      - PCC_A: Arousal 的 Pearson 相關係數
      - RMSE_VA: Valence + Arousal 合併後的 RMSE
        (若 is_norm=True，則會依照官方範例做 sqrt(128) 的正規化)
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PCC_V, PCC_A, RMSE_VA for DimABSA2026 Track A Subtask 1"
    )
    parser.add_argument(
        "--gold_json",
        required=True,
        help="Path to gold JSON (e.g., dev.json with Valence/Arousal)",
    )
    parser.add_argument(
        "--pred_json",
        required=True,
        help="Path to prediction JSON (e.g., predictions.json from test.py)",
    )
    parser.add_argument(
        "--norm_rmse",
        action="store_true",
        help="If set, compute normalized RMSE_VA (divide by sqrt(128)).",
    )
    args = parser.parse_args()

    # 讀取金標與預測
    with open(args.gold_json, "r", encoding="utf-8") as f:
        gold_records = json.load(f)
    with open(args.pred_json, "r", encoding="utf-8") as f:
        pred_records = json.load(f)

    gold_df = pd.DataFrame(gold_records)
    pred_df = pd.DataFrame(pred_records)

    required_cols = ["ID", "Aspect", "Valence", "Arousal"]
    for col in required_cols:
        if col not in gold_df.columns:
            raise ValueError(f"gold_json missing column: {col}")
        if col not in pred_df.columns:
            raise ValueError(f"pred_json missing column: {col}")

    # 只留必要欄位，避免 merge 產生太多重複欄位
    gold_df = gold_df[["ID", "Aspect", "Valence", "Arousal"]].copy()
    pred_df = pred_df[["ID", "Aspect", "Valence", "Arousal"]].copy()

    # 以 ID + Aspect 做對齊
    merged = pd.merge(
        gold_df,
        pred_df,
        on=["ID", "Aspect"],
        suffixes=("_gold", "_pred"),
        validate="one_to_one",  # 若有重複 key 會直接丟錯，方便排查
    )

    print(f"[evaluation] merged examples: {len(merged)}")

    gold_v = merged["Valence_gold"].to_numpy()
    gold_a = merged["Arousal_gold"].to_numpy()
    pred_v = merged["Valence_pred"].to_numpy()
    pred_a = merged["Arousal_pred"].to_numpy()

    metrics = evaluate_predictions_task1(
        pred_a=pred_a,
        pred_v=pred_v,
        gold_a=gold_a,
        gold_v=gold_v,
        is_norm=args.norm_rmse,
    )

    print("===== Evaluation Result =====")
    print(f"PCC_V   : {metrics['PCC_V']:.6f}")
    print(f"PCC_A   : {metrics['PCC_A']:.6f}")
    print(f"RMSE_VA : {metrics['RMSE_VA']:.6f}")
    if args.norm_rmse:
        print("(RMSE_VA is normalized by sqrt(128))")


if __name__ == "__main__":
    main()
