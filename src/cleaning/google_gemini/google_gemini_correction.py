#!/usr/bin/env python
# coding: utf-8

import sys
import os
import re
import json
import time
import asyncio
import pandas as pd
import google.generativeai as genai

# -----------------------------------------------------------------
# 1. Google Gemini API Key
# -----------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY not found. Please set it in the environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)


# -----------------------------------------------------------------
# 2. Read CSV / Error File
# -----------------------------------------------------------------
def load_csv(file_path: str) -> pd.DataFrame:
    """Load the original CSV file."""
    return pd.read_csv(file_path)


def load_error_dict(error_file: str) -> dict:
    """
    Load the error detection file (e.g., gemini_detections.json).
    Format: { "global row number as str": { column name: "error description", ... }, ... }
    """
    with open(error_file, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------
# 3. Split the error dictionary into batches of 20 rows
# -----------------------------------------------------------------
def batch_error_lines(errors_dict: dict, batch_size: int = 20):
    """
    Split error_dict (global row number -> {column -> error description})
    into a list of [(row_str, col_errors), ...] and yield up to 20 rows at a time.
    """
    items = list(errors_dict.items())  # [(row_str, {column -> error description}), ...]
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# -----------------------------------------------------------------
# 4. Construct the Prompt for correction
# -----------------------------------------------------------------
def build_fix_prompt(df: pd.DataFrame, error_batch: list) -> str:
    """
    error_batch: [(row_str, { column: "error description" }), ...] (up to 20 rows)
    df: original data DataFrame (global row index)

    We inject known_stats into the prompt to instruct the model on how to fill missing values,
    or otherwise correct data.
    """

    known_stats = {
        "default_num": 0.055,
        "ipa_num": 0.06,
        "stout_num": 0.065,
        "cities": ["Chicago", "New York", "San Diego", "Portland", "Austin", "Denver", "Seattle", "Houston", "Los Angeles"],
        "states": ["CA", "NY", "TX", "IL", "WA", "OR", "CO", "FL", "MI", "NC", "IN", "AZ", "WI"]
    }

    stats_json = json.dumps(known_stats, indent=2)

    instructions_list = []
    for (row_str, col_errors) in error_batch:
        row_idx = int(row_str)
        if row_idx < 0 or row_idx >= len(df):
            continue
        row_data = df.iloc[row_idx].to_dict()

        instructions_list.append({
            "global_row_index": row_str,
            "original_data": row_data,
            "errors": col_errors
        })

    batch_json = json.dumps(instructions_list, indent=2, ensure_ascii=False)

    # 在提示中更加严格地要求：每个出错列都必须给出修正值。
    prompt = f"""
We have some 'known_stats' for filling missing or invalid values:
{stats_json}

You are a data cleaning expert. For each row in this batch:
1. ONLY fix the columns listed in 'errors'. Do not touch columns with no errors.
2. If a column has a 'missing value' or '缺失值' error, you MUST fill it. 
   - For numeric columns, try to infer from known_stats or use 'default_num' if you're unsure.
   - For city/state, pick a likely option from known_stats if none is obvious from context.
   - Do not leave any missing value unfilled.
3. If a numeric column is missing or invalid, do NOT just fill with 0.0 blindly. Infer from known_stats or use a sensible default/mean.
4. If city/state is missing or invalid, try to infer from known_stats or data context.
5. You must NOT omit any column listed in 'errors'. If a column is in 'errors', you must provide a corrected value for it.
6. Do NOT output disclaimers or extra text. Return a valid JSON that starts with {{ and ends with }}.
7. JSON format:
{{
  "global_row_number": {{
    "column_name": "corrected_value",
    ...
  }},
  "another_global_row_number": {{
    ...
  }}
}}

Below is the batch (max 20 lines) in JSON:

{batch_json}

Now return the corrected JSON object only.
"""
    return prompt.strip()


# -----------------------------------------------------------------
# 5. Call Gemini
# -----------------------------------------------------------------
def call_gemini(prompt: str) -> str:
    """Use gemini-2.0-flash to get the raw text response."""
    model = genai.GenerativeModel(model_name='gemini-2.0-flash')
    response = model.generate_content(prompt)
    raw_text = response.text if (response and response.text) else ""
    return raw_text.strip()


# -----------------------------------------------------------------
# 6. Process corrections in batches
# -----------------------------------------------------------------
async def run_batch_corrections(df: pd.DataFrame, errors_dict: dict, batch_size: int = 20):
    """
    Return a list of raw_corrections:
    [
      {
        "batch_num": 0,
        "error_lines": [...],
        "raw_response": "... (maybe containing code blocks, etc.)"
      },
      ...
    ]
    """
    items_generator = batch_error_lines(errors_dict, batch_size=batch_size)
    all_batches = list(items_generator)
    total_batches = len(all_batches)

    raw_corrections = []
    batch_times = []

    for batch_num, error_batch in enumerate(all_batches):
        t0 = time.time()
        print(f"\n🚀 Processing batch {batch_num}/{total_batches-1}, with {len(error_batch)} lines...")

        prompt = build_fix_prompt(df, error_batch)
        print(f"⏳ Sending batch {batch_num} to Gemini...")

        raw_text = call_gemini(prompt)
        raw_corrections.append({
            "batch_num": batch_num,
            "error_lines": error_batch,  # 用于后面 fallback 检查
            "raw_response": raw_text
        })

        cost = time.time() - t0
        batch_times.append(cost)
        print(f"✅ Batch {batch_num} done in {cost:.2f}s (response length={len(raw_text)})")

        # estimate
        avg_t = sum(batch_times) / len(batch_times)
        remain = total_batches - (batch_num + 1)
        est_sec = remain * avg_t
        mm, ss = divmod(est_sec, 60)
        print(f"🕒 Estimated time left: {int(mm)}m {int(ss)}s")

        # wait a bit
        await asyncio.sleep(2)

    return raw_corrections


# -----------------------------------------------------------------
# 7. Parse raw_gemini_correction.json => final_corrections
# -----------------------------------------------------------------
def parse_raw_corrections(raw_file: str) -> dict:
    """
    read raw_corrections (a list), remove code blocks from raw_response, parse as JSON,
    then merge => { row_str: {col: corrected_value} }
    """
    try:
        with open(raw_file, "r", encoding="utf-8") as f:
            data_list = json.load(f)  # [{"batch_num":..., "error_lines":..., "raw_response":..., ...}, ...]
    except Exception as e:
        print(f"❌ Failed to read {raw_file}: {e}")
        return {}

    final_corrections = {}
    for batch_item in data_list:
        raw_txt = batch_item.get("raw_response", "")
        # Remove possible Markdown code fences
        cleaned = re.sub(r"```(json)?", "", raw_txt)
        cleaned = re.sub(r"```", "", cleaned).strip()

        # 首先尝试解析大模型返回的 JSON
        partial_dict = {}
        try:
            corr_obj = json.loads(cleaned)
            if not isinstance(corr_obj, dict):
                raise ValueError("JSON must be a dictionary")
            # 将解析结果合并到 partial_dict
            for row_str, row_fixes in corr_obj.items():
                row_str = str(row_str)
                if row_str not in partial_dict:
                    partial_dict[row_str] = {}
                for col, val in row_fixes.items():
                    partial_dict[row_str][col] = val
        except json.JSONDecodeError as ex:
            print(f"❌ JSON parse error in batch {batch_item.get('batch_num','?')}: {ex}")

        # ========== 关键：补漏检查 / fallback 逻辑 ==========
        error_lines = batch_item.get("error_lines", [])
        partial_dict = enforce_corrections_for_all_errors(error_lines, partial_dict)

        # 把本批次修正结果合并进 final_corrections
        for row_str, colvals in partial_dict.items():
            if row_str not in final_corrections:
                final_corrections[row_str] = {}
            for col, val in colvals.items():
                final_corrections[row_str][col] = val

    return final_corrections


def enforce_corrections_for_all_errors(error_lines, partial_dict):
    """
    检查大模型输出（partial_dict）是否缺漏了任何错误列：
    如果某个 (row_str, col) 本来在 errors 里，但模型没给修正值，
    则使用 fallback 填充。
    """
    # 可复用和 prompt 中一致的 known_stats
    known_stats = {
        "default_num": 0.055,
        "ipa_num": 0.06,
        "stout_num": 0.065,
        "cities": ["Chicago", "New York", "San Diego", "Portland", "Austin", "Denver", "Seattle", "Houston", "Los Angeles"],
        "states": ["CA", "NY", "TX", "IL", "WA", "OR", "CO", "FL", "MI", "NC", "IN", "AZ", "WI"]
    }

    for (row_str, col_errors) in error_lines:
        row_str = str(row_str)
        # 如果 partial_dict 不包含该行，先创建空字典
        if row_str not in partial_dict:
            partial_dict[row_str] = {}
        # 检查每个错误列是否存在
        for col, err_desc in col_errors.items():
            if col not in partial_dict[row_str]:
                # fallback 补偿
                partial_dict[row_str][col] = fallback_value(col, err_desc, known_stats)
    return partial_dict


def fallback_value(col, err_desc, known_stats):
    """
    如果模型遗漏某列的修正值，则这里给一个缺省值。
    简单逻辑演示：
      1) 如果列名包含 city => 返回 known_stats['cities'][0]
      2) 如果列名包含 state => 返回 known_stats['states'][0]
      3) 否则一律用 known_stats['default_num']
    你也可以做更复杂的推断。
    """
    lower_col = col.lower()
    if "city" in lower_col:
        return known_stats["cities"][0]  # "Chicago"
    if "state" in lower_col:
        return known_stats["states"][0]  # "CA"
    # 数值型列可用 default_num
    return known_stats["default_num"]


# -----------------------------------------------------------------
# 8. Main
# -----------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python gemini_correction.py <csv_file> <error_json>")
        sys.exit(1)

    csv_file = sys.argv[1]
    error_json = sys.argv[2]

    # load data
    df = load_csv(csv_file)
    errors_dict = load_error_dict(error_json)

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    start_time = time.time()
    # run batch corrections
    raw_corrections = asyncio.run(run_batch_corrections(df, errors_dict, batch_size=20))

    # save raw
    with open("raw_gemini_correction.json","w",encoding="utf-8") as f:
        json.dump(raw_corrections, f, indent=4, ensure_ascii=False)
    print("\n✅ raw_gemini_correction.json saved.")

    # parse => gemini_corrections.json
    final_corrections = parse_raw_corrections("raw_gemini_correction.json")
    with open("gemini_corrections.json","w",encoding="utf-8") as f:
        json.dump(final_corrections, f, indent=4, ensure_ascii=False)

    cost = time.time() - start_time
    print(f"\n✅ Done in {cost:.2f} seconds. Corrections in gemini_corrections.json")
