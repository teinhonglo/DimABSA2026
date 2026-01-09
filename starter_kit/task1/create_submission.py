#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import zipfile


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a Codabench-style submission zip from pred_*.jsonl files "
            "in the given output_dir."
        )
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory that contains pred_*.jsonl produced by test.py",
    )
    parser.add_argument(
        "--subtask",
        default="subtask_1",
        help="Subtask folder name inside the zip (default: subtask_1)",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    subtask_dir = os.path.join(output_dir, args.subtask)
    os.makedirs(subtask_dir, exist_ok=True)

    # 找出所有 pred_*.jsonl，支援多個 lang/domain 同時打包
    pred_files = [
        f for f in os.listdir(output_dir)
        if f.startswith("pred_") and f.endswith(".jsonl")
    ]
    if not pred_files:
        raise RuntimeError(f"No pred_*.jsonl files found in {output_dir}")

    # 複製 prediction 檔案到 subtask 資料夾
    for fname in pred_files:
        src = os.path.join(output_dir, fname)
        dst = os.path.join(subtask_dir, fname)
        shutil.copy(src, dst)
        print(f"[create_submission] Copied {src} -> {dst}")

    # 建立 zip 檔，結構為:
    #   subtask_1.zip
    #     └── subtask_1/
    #           pred_*.jsonl
    zip_path = os.path.join(output_dir, f"{args.subtask}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(subtask_dir):
            for file in files:
                full_path = os.path.join(root, file)
                # 相對於 output_dir 的路徑寫進 zip 裡
                arcname = os.path.relpath(full_path, output_dir)
                zf.write(full_path, arcname)

    print(f"[create_submission] Created submission zip: {zip_path}")


if __name__ == "__main__":
    main()
