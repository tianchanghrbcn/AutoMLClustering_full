#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

def main():
    # 1) 解析命令行参数
    parser = argparse.ArgumentParser(description="Calculate correlations between two sets of column indices.")
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument("--output", "-o", default="results.csv", help="Path to output CSV (default: results.csv).")
    args = parser.parse_args()

    # 2) 读取CSV文件为DataFrame
    df = pd.read_csv(args.csv_file)

    # 3) 指定左右两侧列的索引
    #    这里按你的需求，左侧列索引是 6~10 (含 6,7,8,9,10)
    #    右侧列索引: 11,14,17,20,23,26,29,32,35,38,39,42,45,46
    left_indices  = [6, 7, 8, 9, 10]
    right_indices = [11,14,17,20,23,26,29,32,35,38,39,42,45,46]

    # 4) 将索引转换为实际列名（DataFrame.columns 是从0开始的列表）
    left_cols  = [df.columns[i] for i in left_indices]
    right_cols = [df.columns[i] for i in right_indices]

    # 5) 循环计算左右每一列的相关系数
    results = []
    for colA in left_cols:
        for colB in right_cols:
            # 只对数值型列进行相关度计算
            if pd.api.types.is_numeric_dtype(df[colA]) and pd.api.types.is_numeric_dtype(df[colB]):
                corr_val = df[colA].corr(df[colB])  # 皮尔逊相关系数
                results.append({
                    "colA": colA,
                    "colB": colB,
                    "correlation": corr_val
                })

    # 6) 将结果转换为DataFrame，并按相关系数从高到低排序
    corr_df = pd.DataFrame(results)
    corr_df.sort_values(by="correlation", ascending=False, inplace=True)

    # 如果你想按绝对值排序，可改为：
    # corr_df = corr_df.reindex(corr_df['correlation'].abs().sort_values(ascending=False).index)

    # 7) 输出到CSV
    output_path = args.output
    corr_df.to_csv(output_path, index=False)
    print(f"[INFO] Correlation results (rows={len(corr_df)}) written to {output_path}")

if __name__ == "__main__":
    main()
