import pandas as pd
import os

# 1. 定义要处理的文件列表（如 beers, flights, hospital, rayyan）
datasets = ["beers", "flights", "hospital", "rayyan"]

for ds in datasets:
    cleaning_file = f"../../../results/analysis_results/{ds}_cleaning.xlsx"
    cluster_file  = f"../../../results/analysis_results/{ds}_cluster.xlsx"

    # 2. 读取 Excel 表
    df_cleaning = pd.read_excel(cleaning_file)
    df_cluster  = pd.read_excel(cluster_file)

    # 3. 合并：指定键字段(若列名不一致，需要事先 rename)
    #   how='inner'：只保留能匹配到双方键的行
    merge_keys = ["task_name","num","dataset_id","error_rate","cleaning_method"]
    df_merged = pd.merge(df_cleaning, df_cluster, how='inner', on=merge_keys)

    # 4. 输出合并后的文件
    out_filename = f"../../../results/analysis_results/{ds}_summary.xlsx"
    df_merged.to_excel(out_filename, index=False)
    print(f"[INFO] Merged for {ds}: {out_filename} (rows={len(df_merged)})")

