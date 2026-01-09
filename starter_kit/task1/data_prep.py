#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import List, Dict

import pandas as pd
import requests
from sklearn.model_selection import train_test_split


def load_jsonl_url(url: str) -> List[Dict]:
    """Download a JSONL file from a URL and return a list of dicts."""
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def jsonl_to_df(data: List[Dict]) -> pd.DataFrame:
    """
    Convert the raw JSON list from DimABSA2026 into a flat DataFrame with columns:
    ID, Text, Aspect, Valence, Arousal
    """
    if not data:
        raise ValueError("Empty data passed to jsonl_to_df")

    first = data[0]

    if "Quadruplet" in first:
        df = pd.json_normalize(data, "Quadruplet", ["ID", "Text"])
        df[["Valence", "Arousal"]] = df["VA"].str.split("#", expand=True).astype(float)
        df = df.drop(columns=["VA", "Category", "Opinion"], errors="ignore")
        df = df.drop_duplicates(subset=["ID", "Aspect"], keep="first")

    elif "Triplet" in first:
        df = pd.json_normalize(data, "Triplet", ["ID", "Text"])
        df[["Valence", "Arousal"]] = df["VA"].str.split("#", expand=True).astype(float)
        df = df.drop(columns=["VA", "Opinion"], errors="ignore")
        df = df.drop_duplicates(subset=["ID", "Aspect"], keep="first")

    elif "Aspect_VA" in first:
        df = pd.json_normalize(data, "Aspect_VA", ["ID", "Text"])
        df = df.rename(columns={df.columns[0]: "Aspect"})
        df[["Valence", "Arousal"]] = df["VA"].str.split("#", expand=True).astype(float)
        df = df.drop_duplicates(subset=["ID", "Aspect"], keep="first")

    elif "Aspect" in first:
        df = pd.json_normalize(data, "Aspect", ["ID", "Text"])
        df = df.rename(columns={df.columns[0]: "Aspect"})
        # unlabeled set: fill dummy scores with 0
        df["Valence"] = 0.0
        df["Arousal"] = 0.0
        df = df.drop_duplicates(subset=["ID", "Aspect"], keep="first")

    else:
        raise ValueError(
            "Invalid format: must include 'Quadruplet', 'Triplet', 'Aspect_VA', or 'Aspect'"
        )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare train/dev/test JSON for DimABSA2026 Track A, Subtask 1"
    )
    parser.add_argument("--out_data_dir", required=True, help="Directory to write train/dev/test.json")
    parser.add_argument("--lang", default="eng", help="3-letter language code, e.g. eng, zho")
    parser.add_argument("--domain", default="laptop", help="Domain name, e.g. laptop, restaurant")
    parser.add_argument("--subtask", default="subtask_1", help="Subtask name (default: subtask_1)")
    parser.add_argument("--task", default="task1", help="Task name (default: task1)")
    parser.add_argument("--dev_size", type=float, default=0.1, help="Proportion of train used as dev (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/dev split")
    args = parser.parse_args()

    os.makedirs(args.out_data_dir, exist_ok=True)

    base_url = (
        "https://raw.githubusercontent.com/DimABSA/DimABSA2026/"
        "refs/heads/main/task-dataset/track_a"
    )

    train_url = (
        f"{base_url}/{args.subtask}/{args.lang}/"
        f"{args.lang}_{args.domain}_train_alltasks.jsonl"
    )
    predict_url = (
        f"{base_url}/{args.subtask}/{args.lang}/"
        f"{args.lang}_{args.domain}_dev_{args.task}.jsonl"
    )

    print(f"[data_prep] Downloading train from: {train_url}")
    train_raw = load_jsonl_url(train_url)

    print(f"[data_prep] Downloading predict(dev/test) from: {predict_url}")
    predict_raw = load_jsonl_url(predict_url)

    train_df = jsonl_to_df(train_raw)
    predict_df = jsonl_to_df(predict_raw)

    print(f"[data_prep] Train size before split: {len(train_df)}")
    train_df, dev_df = train_test_split(
        train_df, test_size=args.dev_size, random_state=args.seed
    )
    print(f"[data_prep] Train size after split: {len(train_df)}, Dev size: {len(dev_df)}")
    print(f"[data_prep] Test size (unlabeled dev set): {len(predict_df)}")

    splits = {
        "train": train_df,
        "dev": dev_df,
        "test": predict_df,
    }

    for name, df in splits.items():
        path = os.path.join(args.out_data_dir, f"{name}.json")
        records = df.to_dict(orient="records")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[data_prep] Saved {name} to {path} ({len(df)} examples)")

    meta = {
        "subtask": args.subtask,
        "task": args.task,
        "lang": args.lang,
        "domain": args.domain,
        "dev_size": args.dev_size,
        "seed": args.seed,
        "train_url": train_url,
        "predict_url": predict_url,
    }
    meta_path = os.path.join(args.out_data_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[data_prep] Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
