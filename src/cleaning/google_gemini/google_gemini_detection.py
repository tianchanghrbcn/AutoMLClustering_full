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
import google.generativeai as genai

# ---------------------------------------------------------
# 1. 配置 Google Gemini API
# ---------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY not found. Please set it in the environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)


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
        yield i, df.iloc[i: i + batch_size].reset_index(drop=True)


# ---------------------------------------------------------
# 3. Prompt 生成函数
# ---------------------------------------------------------
def create_prompt(df: pd.DataFrame) -> str:
    """Generate a structured prompt for Gemini based on the DataFrame sample."""
    column_names = df.columns.tolist()
    sample_data = df.to_dict(orient="records")  # convert a batch of rows to a list of dicts

    # 在 Prompt 中明确说明只返回 JSON
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
# 4. 向 Gemini 发送请求的函数
# ---------------------------------------------------------
def gemini_request(
        df: pd.DataFrame,
        batch_start_idx: int,
        batch_num: int,
        requests_history: list,
        max_retries: int = 3
):
    """Send prompt to Gemini 2.0 Pro and parse JSON response."""
    prompt = create_prompt(df)
    request_time = datetime.datetime.now().isoformat()

    for attempt in range(max_retries):
        try:
            # 1) 指定 Gemini 2.0 Pro (可能名称随版本变化)
            model = genai.GenerativeModel(model_name='gemini-2.0-flash')
            response = model.generate_content(prompt)

            # 2) 取到响应文本
            raw_response_text = response.text.strip() if response and response.text else ""

            # 3) 构建交互记录
            interaction_record = {
                "batch_num": batch_num,
                "attempt": attempt + 1,
                "request_time": request_time,
                "prompt": prompt,
                "gemini_raw_response": raw_response_text
            }

            # 4) 空响应处理
            if not raw_response_text:
                interaction_record["parsed_issues"] = {}
                interaction_record["error"] = "Empty response"
                requests_history.append(interaction_record)
                return raw_response_text, {}

            # 5) 尝试解析 JSON
            try:
                issues = json.loads(raw_response_text)
                if not isinstance(issues, dict):
                    raise ValueError("Returned JSON is not a dictionary")
            except (json.JSONDecodeError, ValueError) as e:
                interaction_record["parsed_issues"] = {}
                interaction_record["error"] = f"JSON parsing failed: {e}"
                requests_history.append(interaction_record)
                return raw_response_text, {}

            # 6) 处理行号修正（如果模型返回 "row_index"）
            corrected_issues = {}
            for row_index_str, row_errors in issues.items():
                try:
                    numeric_row = int(row_index_str)
                    corrected_index = batch_start_idx + numeric_row
                    corrected_issues[str(corrected_index)] = row_errors
                except ValueError:
                    # 如果键不是数字，则保留原键
                    corrected_issues[row_index_str] = row_errors

            # 7) 记录并返回
            interaction_record["parsed_issues"] = corrected_issues
            requests_history.append(interaction_record)
            return raw_response_text, corrected_issues

        except Exception as e:
            print(f"⚠️ Gemini request failed [batch {batch_num} attempt {attempt + 1}/{max_retries}]: {e}")
            time.sleep(5)

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
    """Parse the raw responses file to aggregate final issues."""
    try:
        with open(raw_responses_file, "r", encoding="utf-8") as f:
            raw_responses = json.load(f)

        consolidated_issues = {}
        for batch in raw_responses:
            try:
                response_text = batch.get("raw_response", "").strip()
                if not response_text:
                    print(f"⚠️ Skipping empty response: batch {batch['batch_num']}")
                    continue

                cleaned_text = re.sub(r"```(json)?\s*|\s*```", "", response_text).strip()

                try:
                    issues = json.loads(cleaned_text)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing failed for batch {batch['batch_num']}: {e}")
                    continue

                if isinstance(issues, dict) and issues:
                    consolidated_issues.update(issues)
                else:
                    print(f"⚠️ batch {batch['batch_num']}: data is empty or invalid.")
            except Exception as e:
                print(f"❌ Unexpected error in batch {batch['batch_num']}: {e}")

        return consolidated_issues

    except Exception as e:
        print(f"❌ Failed to read {raw_responses_file}: {e}")
        return {}


# ---------------------------------------------------------
# 6. 主流程：批次分析 + 存储结果 + 解析汇总
# ---------------------------------------------------------
async def detect_with_gemini(df: pd.DataFrame):
    requests_history = []
    consolidated_issues = {}
    all_raw_responses = []

    for batch_num, (batch_start_idx, batch_df) in enumerate(batch_processing(df, batch_size=50)):
        print(f"🚀 Processing batch {batch_num} (rows {batch_start_idx} to {batch_start_idx + len(batch_df) - 1})...")
        raw_response, issues = gemini_request(batch_df, batch_start_idx, batch_num, requests_history)
        all_raw_responses.append({
            "batch_num": batch_num,
            "raw_response": raw_response
        })
        consolidated_issues.update(issues)

    save_json(all_raw_responses, "raw_gemini_responses.json")
    return consolidated_issues


# ---------------------------------------------------------
# 7. 主入口
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python google_gemini_detection.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]  # 从命令行获取 CSV 文件路径

    start_time = time.time()

    # 读取指定路径的 CSV
    df = load_csv(csv_path)

    # 针对 Windows 的事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    final_issues = asyncio.run(detect_with_gemini(df))

    # 二次解析 raw_gemini_responses.json 并合并
    final_parsed_issues = parse_raw_responses("raw_gemini_responses.json")
    if final_parsed_issues:
        final_issues.update(final_parsed_issues)

    output_file = "gemini_detections.json"
    save_json(final_issues, output_file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"⏳ Program execution time: {elapsed_time:.2f} seconds")
