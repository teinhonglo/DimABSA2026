#!/usr/bin/env python
# merge_domains.py
# 將多個 (lang, domain) 的 train.json / dev.json 合併為一個新的資料夾
# train = sum over all langs,domains train
# dev   = sum over all langs,domains dev
# test 不處理

import argparse
import json
import os
from typing import List, Any


def load_json(path: str) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: List[Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Merge train/dev json from multiple (lang, domain) folders for DimABSA Track A."
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language prefix(es), comma-separated, e.g. 'eng' or 'eng,chn'",
    )
    parser.add_argument(
        "--domains",
        type=str,
        required=True,
        help="Comma-separated domain list to merge, e.g. 'laptop,restaurant'",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory where {lang}_{domain} folders are located",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="multi",
        help="Output directory. Merged train/dev will be saved in {out_dir}",
    )
    args = parser.parse_args()

    # 支援多個 lang, e.g. "eng,chn"
    langs = [l.strip() for l in args.lang.split(",") if l.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    if not langs:
        raise ValueError("No valid langs provided. Use e.g. --lang eng,chn")
    if not domains:
        raise ValueError("No valid domains provided. Use e.g. --domains laptop,restaurant")

    print(f"[merge_domains] langs       = {langs}")
    print(f"[merge_domains] domains     = {domains}")
    print(f"[merge_domains] data_root   = {args.data_root}")
    print(f"[merge_domains] out_dir     = {args.out_dir}")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"[merge_domains] output dir  = {out_dir}")

    # 只合併 train / dev
    for split in ["train", "dev"]:
        merged: List[Any] = []
        print(f"[merge_domains] ===== merging {split}.json =====")

        for lang in {l for l in langs}:  # 去重，避免重複
            for dom in domains:
                src_dir = os.path.join(args.data_root, f"{lang}_{dom}")
                src_path = os.path.join(src_dir, f"{split}.json")
                if not os.path.exists(src_path):
                    print(
                        f"[merge_domains][WARN] {src_path} not found, "
                        f"skip this (lang={lang}, domain={dom}) for {split}."
                    )
                    continue

                print(f"[merge_domains] loading {src_path}")
                data = load_json(src_path)
                if not isinstance(data, list):
                    raise ValueError(f"{src_path} is not a list of records.")

                merged.extend(data)
                print(
                    f"[merge_domains] {lang}_{dom} {split}: "
                    f"+{len(data)} samples (total={len(merged)})"
                )

        out_path = os.path.join(out_dir, f"{split}.json")
        print(f"[merge_domains] saving merged {split} to {out_path} (total={len(merged)})")
        save_json(merged, out_path)

    print("[merge_domains] done.")


if __name__ == "__main__":
    main()
