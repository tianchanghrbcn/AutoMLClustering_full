#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pandas as pd
import json
import re
import asyncio
import time
import datetime
import requests

# ---------------------------------------------------------
# 1. 配置 Google Gemini API Key
# ---------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY not found. Please set it in the environment variables.")

# 你要调用的模型名称，如 "models/gemini-2.0-flash" 或 "models/text-bison-001"
MODEL_NAME = "models/gemini-2.0-flash"

# ---------------------------------------------------------
# 2. CSV 读取与处理
# ---------------------------------------------------------
def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV into a pandas DataFrame."""
    df = pd.read_csv(file_path)
    return df

def batch_processing(df: pd.DataFrame, batch_size: int = 50):
    """Yield consecutive 50-row batches from the DataFrame."""
    for i in range(0, len(df), batch_size):
        yield i, df.iloc[i : i + batch_size].reset_index(drop=True)

# ---------------------------------------------------------
# 3. Prompt 生成函数
# ---------------------------------------------------------
def create_prompt(df: pd.DataFrame) -> str:
    """
    Generate a structured prompt for Gemini/PaLM based on the DataFrame batch sample.
    We'll instruct it to only return JSON with detected errors.
    """
    column_names = df.columns.tolist()
    sample_data = df.to_dict(orient="records")  # up to 50 rows in this batch

    prompt = f"""
You are a data analysis expert. Below is a dataset with column names and sample data:

Column Names:
{column_names}

Sample Data (max 50 rows in this batch):
{sample_data}

Please analyze potential errors and return a JSON structure with detected errors per row:

1. Missing Values
2. Formatting Errors (dates, emails, phone numbers)
3. Spelling Errors (names, cities, companies)
4. Outliers in numerical fields
5. Dependency Errors (invalid date logic, mismatched country-capital, etc.)

IMPORTANT:
- You must return a single JSON object with the exact format:
{{
  "row_index": {{
    "column_name": "Error description",
    "another_column": "Error description"
  }},
  "another_row_index": {{
    ...
  }}
}}
- If there are no errors, return an empty JSON {{}}.
- Do NOT include explanations, disclaimers, or markdown formatting.
- The JSON must start with {{ and end with }}.
""".strip()

    return prompt

# ---------------------------------------------------------
# 4. 用 REST 调用 PaLM/Gemini 接口
# ---------------------------------------------------------
def gemini_request_via_rest(prompt: str, api_key: str, model: str = MODEL_NAME) -> str:
    """
    Use HTTP POST to the Generative Language API endpoint: generateText.
    We can set temperature, maxOutputTokens, etc. in the payload.
    Returns the raw text output (or empty string on failure).
    """
    url = f"https://generativelanguage.googleapis.com/v1beta2/{model}:generateText?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": {"text": prompt},
        "temperature": 0.2,
        "maxOutputTokens": 256,
        # you can add more parameters: topK, topP, candidateCount, etc.
    }

    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code == 200:
        data = resp.json()
        # Typically: {"candidates": [{"output": "..."}], ...}
        candidates = data.get("candidates", [])
        if candidates:
            return candidates[0].get("output", "").strip()
        else:
            return ""
    else:
        raise RuntimeError(
            f"PaLM/Gemini request failed, status={resp.status_code}, msg={resp.text}"
        )

def gemini_request(
    df: pd.DataFrame,
    batch_start_idx: int,
    batch_num: int,
    requests_history: list,
    max_retries: int = 3
):
    """
    For the given batch DataFrame, create a prompt and call gemini_request_via_rest
    with automatic retries. Return (raw_response_text, parsed_issues).
    """
    prompt = create_prompt(df)
    request_time = datetime.datetime.now().isoformat()

    for attempt in range(max_retries):
        try:
            raw_response_text = gemini_request_via_rest(
                prompt=prompt,
                api_key=GOOGLE_API_KEY,
                model=MODEL_NAME
            )

            interaction_record = {
                "batch_num": batch_num,
                "attempt": attempt + 1,
                "request_time": request_time,
                "prompt": prompt,
                "gemini_raw_response": raw_response_text
            }

            if not raw_response_text:
                # empty response
                interaction_record["parsed_issues"] = {}
                interaction_record["error"] = "Empty response"
                requests_history.append(interaction_record)
                return raw_response_text, {}

            # parse JSON
            try:
                issues = json.loads(raw_response_text)
                if not isinstance(issues, dict):
                    raise ValueError("Returned JSON is not a dictionary")
            except (json.JSONDecodeError, ValueError) as e:
                interaction_record["parsed_issues"] = {}
                interaction_record["error"] = f"JSON parsing failed: {e}"
                requests_history.append(interaction_record)
                return raw_response_text, {}

            # convert row_index from local batch to global index
            corrected_issues = {}
            for row_index_str, row_errors in issues.items():
                try:
                    numeric_row = int(row_index_str)
                    corrected_index = batch_start_idx + numeric_row
                    corrected_issues[str(corrected_index)] = row_errors
                except ValueError:
                    # if not numeric, just keep original
                    corrected_issues[row_index_str] = row_errors

            interaction_record["parsed_issues"] = corrected_issues
            requests_history.append(interaction_record)
            return raw_response_text, corrected_issues

        except Exception as e:
            print(f"⚠️ Gemini request failed [batch {batch_num} attempt {attempt+1}/{max_retries}]: {e}")
            time.sleep(3)

    print(f"❌ [batch {batch_num}] API request failed after {max_retries} retries")
    return "", {}

# ---------------------------------------------------------
# 5. 结果保存、解析与合并
# ---------------------------------------------------------
def save_json(data, output_file: str):
    """Save Python object to a JSON file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Failed to save {output_file}: {e}")

def parse_raw_responses(raw_responses_file: str):
    """
    Parse the raw_responses file and combine final issues.
    Each raw response is presumably a JSON string.
    """
    try:
        with open(raw_responses_file, "r", encoding="utf-8") as f:
            raw_responses = json.load(f)

        consolidated_issues = {}
        for batch_item in raw_responses:
            response_text = batch_item.get("raw_response", "").strip()
            if not response_text:
                print(f"⚠️ Skipping empty response: batch {batch_item['batch_num']}")
                continue

            # remove any fenced code blocks
            cleaned_text = re.sub(r"```(json)?\s*|\s*```", "", response_text).strip()
            try:
                issues = json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed for batch {batch_item['batch_num']}: {e}")
                continue

            if isinstance(issues, dict) and issues:
                consolidated_issues.update(issues)
            else:
                print(f"⚠️ batch {batch_item['batch_num']}: data is empty or invalid.")

        return consolidated_issues

    except Exception as e:
        print(f"❌ Failed to read {raw_responses_file}: {e}")
        return {}

# ---------------------------------------------------------
# 6. 主流程：批次分析 + 存储结果 + 解析汇总
# ---------------------------------------------------------
async def detect_with_gemini(df: pd.DataFrame):
    """
    For the entire CSV, do batch-based detection, save raw responses to 'raw_gemini_responses.json',
    and return a dict of consolidated issues.
    """
    requests_history = []
    consolidated_issues = []
    all_raw_responses = []

    for batch_num, (batch_start_idx, batch_df) in enumerate(batch_processing(df, batch_size=50)):
        print(f"🚀 Processing batch {batch_num} (rows {batch_start_idx} to {batch_start_idx + len(batch_df) - 1})...")
        raw_response, issues = gemini_request(batch_df, batch_start_idx, batch_num, requests_history)
        all_raw_responses.append({
            "batch_num": batch_num,
            "raw_response": raw_response
        })
        if issues:
            # issues 是类似 {"10": {...}, "11": {...}}
            # 不断合并到 consolidated_issues
            for row_str, errinfo in issues.items():
                consolidated_issues.append((row_str, errinfo))

        # sleep a bit to avoid hitting rate limits
        await asyncio.sleep(1)

    # 保存所有原始响应
    save_json(all_raw_responses, "raw_gemini_responses.json")

    # 把 list of (row_str, errinfo) 合并为 dict
    final_issues = {}
    for row_str, errinfo in consolidated_issues:
        if row_str not in final_issues:
            final_issues[row_str] = {}
        # errinfo 是一个 {column_name: "Error desc", ...}
        for col, msg in errinfo.items():
            final_issues[row_str][col] = msg

    return final_issues


# ---------------------------------------------------------
# 7. 主入口
# ---------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python google_gemini_detection.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    start_time = time.time()

    # 读取 CSV
    df = load_csv(csv_path)

    # 针对 Windows 的事件循环策略
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 执行检测
    final_issues = asyncio.run(detect_with_gemini(df))

    # 二次解析 raw_gemini_responses.json 并合并 (可选，如果你的逻辑需要二次解析)
    final_parsed_issues = parse_raw_responses("raw_gemini_responses.json")
    # 把二次解析到的内容合并到 final_issues
    for row_str, col_dict in final_parsed_issues.items():
        if row_str not in final_issues:
            final_issues[row_str] = {}
        for c, desc in col_dict.items():
            final_issues[row_str][c] = desc

    # 导出最终
    output_file = "gemini_detections.json"
    save_json(final_issues, output_file)

    elapsed = time.time() - start_time
    print(f"\n✅ Done. Execution time: {elapsed:.2f} seconds.")
    print(f"   Detected issues saved in {output_file}")
