#!/usr/bin/env python
# translate_mlang2eng_multi.py
# 將多語言的 train.json / dev.json 翻譯成英文並合併為 mlang2eng_multi

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

from openai import OpenAI

# ---------- IO utils ----------

def load_json(path: str) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: List[Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------- OpenAI translation ----------

def translate_one(
    client: OpenAI,
    model: str,
    text: str,
    aspect: str,
    max_retries: int = 3,
    sleep_sec: float = 0.2,
) -> Tuple[str, str]:
    """
    使用 OpenAI 將 (text, aspect) 翻成英文，回傳 (text_en, aspect_en)
    如果已經是英文，模型會原樣保留。
    """
    import json as pyjson

    prompt = (
        "You are a professional translator.\n"
        "The input may be in any language (e.g., Japanese, Russian, Chinese).\n"
        "Translate the following review text and its aspect term into natural English.\n"
        "Preserve sentiment intensity and nuances.\n\n"
        "Return ONLY a JSON object with two string fields:\n"
        '{ "text_en": "...", "aspect_en": "..." }.\n\n'
        f"Text: {text}\n"
        f"Aspect: {aspect}\n"
    )

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=0,
                text={"format": {"type": "text"}},
            )
            # 回傳的是一個 Response 物件，文字在 output[0].content[0].text
            raw = resp.output[0].content[0].text  # type: ignore[attr-defined]
            data = pyjson.loads(raw)
            text_en = data.get("text_en", "").strip()
            aspect_en = data.get("aspect_en", "").strip()

            # 防呆：若模型沒有照規則，就 fallback 成原文
            if not text_en:
                text_en = text
            if not aspect_en:
                aspect_en = aspect

            return text_en, aspect_en

        except Exception as e:
            print(f"[translate_one] error (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                print("[translate_one] give up, use original text/aspect.")
                return text, aspect
            time.sleep(sleep_sec)

    # 理論上不會到這裡
    return text, aspect


# ---------- main pipeline ----------

def list_lang_domain_dirs(data_root: str) -> List[Tuple[str, str, str]]:
    """
    掃描 data_root 下的子資料夾，回傳 (lang, domain, full_path) 列表。
    排除:
      - 非資料夾
      - 名稱中沒有 "_" 的
      - lang == "eng"
      - 聚合用的 mlang_xxx 資料夾
    """
    result: List[Tuple[str, str, str]] = []
    for name in sorted(os.listdir(data_root)):
        full_path = os.path.join(data_root, name)
        if not os.path.isdir(full_path):
            continue
        if name.startswith("mlang"):
            continue
        if "_" not in name:
            continue
        lang, domain = name.split("_", 1)
        if lang == "eng":
            continue
        result.append((lang, domain, full_path))
    return result


def process_split(
    split: str,
    dirs: List[Tuple[str, str, str]],
    client: OpenAI,
    model: str,
) -> List[Dict[str, Any]]:
    """
    處理單一 split (train 或 dev)，把所有非 eng domain 的資料翻譯並合併。
    """
    merged: List[Dict[str, Any]] = []

    print(f"[translate] ===== split = {split} =====")
    for lang, domain, dir_path in dirs:
        src_path = os.path.join(dir_path, f"{split}.json")
        if not os.path.exists(src_path):
            print(f"[translate][WARN] {src_path} not found, skip.")
            continue

        print(f"[translate] loading {src_path}")
        data = load_json(src_path)
        if not isinstance(data, list):
            raise ValueError(f"{src_path} is not a list of records.")

        print(f"[translate] {lang}_{domain} {split}: {len(data)} samples")
        for i, rec in enumerate(data, start=1):
            text = str(rec.get("Text", ""))
            aspect = str(rec.get("Aspect", ""))

            text_en, aspect_en = translate_one(
                client=client,
                model=model,
                text=text,
                aspect=aspect,
            )

            new_rec: Dict[str, Any] = dict(rec)
            # 保留原始文字
            new_rec["Text_orig"] = text
            new_rec["Aspect_orig"] = aspect
            # 覆蓋為英文
            new_rec["Text"] = text_en
            new_rec["Aspect"] = aspect_en
            # 加上來源資訊
            new_rec["Lang"] = lang
            new_rec["Domain"] = domain

            merged.append(new_rec)

            if i % 50 == 0:
                print(
                    f"[translate] {lang}_{domain} {split}: processed {i}/{len(data)} "
                    f"(total merged={len(merged)})"
                )

    print(f"[translate] split={split} total merged={len(merged)}")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Translate non-English DimABSA Track A data to English and merge into mlang2eng_multi."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory containing lang_domain folders, e.g. ./data",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./data/mlang2eng_multi",
        help="Output directory for merged translated data.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model name used for translation.",
    )
    args = parser.parse_args()

    print(f"[main] data_root = {args.data_root}")
    print(f"[main] out_dir   = {args.out_dir}")
    print(f"[main] model     = {args.model}")

    dirs = list_lang_domain_dirs(args.data_root)
    if not dirs:
        raise RuntimeError("No non-English lang_domain folders found.")
    print("[main] will process the following lang_domain folders:")
    for lang, domain, path in dirs:
        print(f"  - {lang}_{domain}: {path}")

    os.makedirs(args.out_dir, exist_ok=True)

    client = OpenAI()  # 需要環境變數 OPENAI_API_KEY

    # train
    train_merged = process_split("train", dirs, client, args.model)
    save_json(train_merged, os.path.join(args.out_dir, "train.json"))

    # dev
    dev_merged = process_split("dev", dirs, client, args.model)
    save_json(dev_merged, os.path.join(args.out_dir, "dev.json"))

    print("[main] done. Saved to:")
    print(f"  {os.path.join(args.out_dir, 'train.json')}")
    print(f"  {os.path.join(args.out_dir, 'dev.json')}")


if __name__ == "__main__":
    main()
