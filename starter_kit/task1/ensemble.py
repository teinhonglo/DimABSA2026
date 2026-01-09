#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import List

import numpy as np
import pandas as pd


def extract_num(s: str) -> int:
    """Extract trailing integer from a string (for sorting IDs)."""
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else -1


def df_to_jsonl(df: pd.DataFrame, out_path: str) -> None:
    """
    Convert a flat DF (ID, Aspect, Valence, Arousal) to submission JSONL format.

    Each line:
        {"ID": ..., "Aspect_VA": [{"Aspect": ..., "VA": "v#a"}, ...]}
    """
    df_sorted = df.sort_values(by="ID", key=lambda x: x.map(extract_num))
    grouped = df_sorted.groupby("ID", sort=False)

    with open(out_path, "w", encoding="utf-8") as f:
        for gid, gdf in grouped:
            record = {
                "ID": gid,
                "Aspect_VA": [],
            }
            for _, row in gdf.iterrows():
                record["Aspect_VA"].append(
                    {
                        "Aspect": row["Aspect"],
                        "VA": f"{row['Valence']:.2f}#{row['Arousal']:.2f}",
                    }
                )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_dirs(comma_separated: str) -> List[str]:
    return [d.strip() for d in comma_separated.split(",") if d.strip()]


def parse_weights(num_models: int, weights_str: str | None) -> np.ndarray:
    if not weights_str:
        w = np.ones(num_models, dtype=float)
    else:
        parts = [p.strip() for p in weights_str.split(",") if p.strip()]
        if len(parts) != num_models:
            raise ValueError(
                f"Number of weights ({len(parts)}) does not match number of models ({num_models})."
            )
        w = np.array([float(p) for p in parts], dtype=float)

    if np.any(w < 0):
        raise ValueError("All weights must be non-negative.")

    if np.all(w == 0):
        raise ValueError("At least one weight must be positive.")

    w = w / w.sum()
    return w


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ensemble multiple model predictions (predictions.json) by weighted averaging "
            "Valence and Arousal, and write ensembled predictions + pred_lang_domain.jsonl."
        )
    )
    parser.add_argument(
        "--model_preddirs",
        required=True,
        help="Comma-separated list of directories, each containing predictions.json",
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Optional comma-separated weights; if empty, use equal weights",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save ensembled predictions",
    )
    parser.add_argument(
        "--lang",
        required=True,
        help="Language code used in pred_{lang}_{domain}.jsonl",
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain name used in pred_{lang}_{domain}.jsonl",
    )
    args = parser.parse_args()

    model_dirs = parse_dirs(args.model_preddirs)
    if not model_dirs:
        raise ValueError("No valid model_preddirs parsed.")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[ensemble] model_preddirs: {model_dirs}")
    print(f"[ensemble] output_dir: {args.output_dir}")
    print(f"[ensemble] lang={args.lang}, domain={args.domain}")

    # 讀取所有 predictions.json
    dfs = []
    for d in model_dirs:
        pred_path = os.path.join(d, "predictions.json")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"predictions.json not found in {d}")
        with open(pred_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        print(f"[ensemble] loaded {len(df)} predictions from {pred_path}")
        dfs.append(df)

    num_models = len(dfs)
    weights = parse_weights(num_models, args.weights)
    print(f"[ensemble] using weights: {weights.tolist()}")

    # 以 (ID, Aspect) 對齊
    base_df = None
    for i, df in enumerate(dfs):
        # 確保必要欄位存在
        for col in ["ID", "Aspect", "Valence", "Arousal"]:
            if col not in df.columns:
                raise ValueError(f"{col} missing in predictions.json from model {i}: {model_dirs[i]}")

        sub = df[["ID", "Aspect", "Valence", "Arousal"]].copy()
        sub = sub.rename(
            columns={
                "Valence": f"Valence_{i}",
                "Arousal": f"Arousal_{i}",
            }
        )

        if base_df is None:
            base_df = sub
        else:
            base_df = pd.merge(
                base_df,
                sub,
                on=["ID", "Aspect"],
                how="inner",
                validate="one_to_one",
            )

    if base_df is None or base_df.empty:
        raise ValueError("No data after merging predictions; check model_preddirs and inputs.")

    print(f"[ensemble] merged rows: {len(base_df)}")

    # 計算加權平均的 Valence / Arousal
    val_cols = [f"Valence_{i}" for i in range(num_models)]
    aro_cols = [f"Arousal_{i}" for i in range(num_models)]

    val_mat = base_df[val_cols].to_numpy(dtype=float)  # (N, M)
    aro_mat = base_df[aro_cols].to_numpy(dtype=float)

    # (N, M) dot (M,) -> (N,)
    val_ens = np.dot(val_mat, weights)
    aro_ens = np.dot(aro_mat, weights)

    base_df["Valence"] = val_ens
    base_df["Arousal"] = aro_ens

    # 從第一個模型拿 Text 欄位（若存在）
    df0 = dfs[0]
    if "Text" in df0.columns:
        text_df = df0[["ID", "Aspect", "Text"]].copy()
        ens_df = pd.merge(
            base_df[["ID", "Aspect", "Valence", "Arousal"]],
            text_df,
            on=["ID", "Aspect"],
            how="left",
        )
        # 調整欄位順序
        ens_df = ens_df[["ID", "Text", "Aspect", "Valence", "Arousal"]]
    else:
        ens_df = base_df[["ID", "Aspect", "Valence", "Arousal"]].copy()

    # 存 ensemble 過的 predictions.json
    out_pred_json = os.path.join(args.output_dir, "predictions.json")
    with open(out_pred_json, "w", encoding="utf-8") as f:
        json.dump(ens_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    print(f"[ensemble] saved ensembled predictions to {out_pred_json}")

    # 存 pred_{lang}_{domain}.jsonl，供 create_submission.py 使用
    out_jsonl = os.path.join(args.output_dir, f"pred_{args.lang}_{args.domain}.jsonl")
    df_to_jsonl(ens_df, out_jsonl)
    print(f"[ensemble] saved ensembled submission jsonl to {out_jsonl}")


if __name__ == "__main__":
    main()
