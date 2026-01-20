#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import model as model_module
from trainer import predict, VADataset


def extract_num(s: str) -> int:
    """Extract trailing integer from a string (for sorting IDs)."""
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else -1


def df_to_jsonl(df: pd.DataFrame, out_path: str) -> None:
    """
    Convert a flat DF (ID, Aspect, Valence, Arousal, Text) to submission JSONL format.

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


def main():
    parser = argparse.ArgumentParser(description="Run inference on test.json with a trained model")
    parser.add_argument("--test_json", required=True, help="Path to test.json produced by data_prep.py")
    parser.add_argument("--model_dir", required=True, help="Directory with best_model.pt and configs")
    parser.add_argument("--output_dir", required=True, help="Directory to store prediction files")
    parser.add_argument("--lang", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--ensemble_alpha", type=float, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    with open(args.test_json, "r", encoding="utf-8") as f:
        test_records = json.load(f)
    test_df = pd.DataFrame(test_records)

    # Load model / tokenizer config
    model_args_path = os.path.join(args.model_dir, "model_args.json")
    if os.path.exists(model_args_path):
        with open(model_args_path, "r", encoding="utf-8") as f:
            model_args = json.load(f)
    else:
        model_args = {}
    
    if args.ensemble_alpha:
        model_args["ensemble_alpha"] = args.ensemble_alpha

    print(f"model_args {model_args}")
    model_class_name = model_args.pop("model_class", "TransformerVARegressor")
    trainer_args_path = os.path.join(args.model_dir, "trainer_args.json")
    trainer_args = {}
    if os.path.exists(trainer_args_path):
        with open(trainer_args_path, "r", encoding="utf-8") as f:
            trainer_args = json.load(f)

    model_name = model_args.get("model_name", "bert-base-multilingual-cased")

    # Prefer tokenizer saved with the experiment; fallback to the original model_name.
    if os.path.isdir(args.model_dir) and os.listdir(args.model_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    batch_size = args.batch_size or int(trainer_args.get("batch_size", 64))
    max_len = args.max_len or int(trainer_args.get("max_len", 128))

    test_dataset = VADataset(test_df, tokenizer, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] Using device: {device}")

    try:
        ModelClass = getattr(model_module, model_class_name)
    except AttributeError:
        raise ValueError(
            f"[test] Unknown model_class='{model_class_name}' in model_args.json. "
            f"Please check your config or model.py."
        )

    model = ModelClass(**model_args).to(device)

    model_path = os.path.join(args.model_dir, "best_model.pt")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Predict
    pred_v, pred_a = predict(model, test_loader, device)

    test_df["Valence"] = pred_v
    test_df["Arousal"] = pred_a

    # Save raw predictions (flat JSON)
    raw_pred_path = os.path.join(args.output_dir, "predictions.json")
    with open(raw_pred_path, "w", encoding="utf-8") as f:
        json.dump(test_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    print(f"[test] Saved raw predictions to {raw_pred_path}")

    # Try to recover lang/domain from meta.json (copied during train.py)
    meta_path = os.path.join(args.model_dir, "meta.json")
    lang = args.lang
    domain = args.domain
    
    # Save in official submission format for this lang/domain.
    submit_path = os.path.join(args.output_dir, f"pred_{lang}_{domain}.jsonl")
    df_to_jsonl(test_df, submit_path)
    print(f"[test] Saved submission-style predictions to {submit_path}")


if __name__ == "__main__":
    main()
