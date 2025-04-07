import os
import pandas as pd
import numpy as np

def main():
    # 要处理的数据集列表
    datasets = ["beers","flights","hospital","rayyan"]

    for ds in datasets:
        # 1) 构建文件路径
        cleaning_file = f"../../../results/analysis_results/{ds}_cleaning.xlsx"
        cluster_file  = f"../../../results/analysis_results/{ds}_cluster.xlsx"
        if not os.path.isfile(cleaning_file) or not os.path.isfile(cluster_file):
            print(f"[WARN] Missing either {cleaning_file} or {cluster_file}, skip {ds}.")
            continue

        # 2) 读取
        df_cleaning = pd.read_excel(cleaning_file)
        df_cluster  = pd.read_excel(cluster_file)

        # 3) 合并 => df_merged
        #    假设有以下键字段
        merge_keys = ["task_name","num","dataset_id","error_rate","cleaning_method"]
        df_merged = pd.merge(df_cleaning, df_cluster, how='inner', on=merge_keys)

        print(f"[INFO] Merged {ds}: df_merged rows={len(df_merged)}")

        # 4) 找GroundTruth行, 并重命名 Sil, DB, Combined -> sil_gt, db_gt, comb_gt
        df_gt = df_merged.loc[df_merged["cleaning_method"]=="GroundTruth"].copy()
        if df_gt.empty:
            print(f"[WARN] No GroundTruth row in {ds}, skip relative calc.")
            # 不妨把df_merged直接输出?
            out_nogt = f"../../../results/analysis_results/{ds}_summary_rel.xlsx"
            df_merged.to_excel(out_nogt, index=False)
            continue

        # 重命名
        df_gt = df_gt.rename(columns={
            "Silhouette Score":"sil_gt",
            "Davies-Bouldin Score":"db_gt",
            "Combined Score":"comb_gt"
        })

        # 5) 指定要匹配的列(除了 cleaning_method, parameters)
        #    仅当这些都一致才认为“同一行” => 用 GT 行做对比
        group_cols = [
            "task_name","num","dataset_id","error_rate",
            "m_x","n_x","anomaly_x","missing_x","cluster_method"
        ]
        # 如果df里没有m_x等列，可根据实际文件调整

        # 只保留 group_cols + [sil_gt, db_gt, comb_gt]
        keep_cols = group_cols + ["sil_gt","db_gt","comb_gt"]
        df_gt = df_gt[keep_cols].copy()

        # 6) 与 df_merged做 left join => df_joined
        #    => 用 group_cols 匹配 => 这样即便 parameters不同也能找到同一组
        df_joined = pd.merge(
            df_merged, df_gt,
            how="left",
            on=group_cols,
            suffixes=("","_g")
        )

        # 7) 计算相对值(若 ground=0 => NaN)
        #    Sil_relative = row["Silhouette Score"]/row["sil_gt"]
        #    DB_relative  = row["db_gt"]/row["Davies-Bouldin Score"]
        #    Comb_relative= row["Combined Score"]/row["comb_gt"]
        df_joined["Sil_relative"] = np.where(
            df_joined["sil_gt"].notnull() & (df_joined["sil_gt"]!=0),
            df_joined["Silhouette Score"] / df_joined["sil_gt"],
            np.nan
        )
        df_joined["DB_relative"] = np.where(
            df_joined["db_gt"].notnull() & (df_joined["db_gt"]!=0),
            df_joined["db_gt"] / df_joined["Davies-Bouldin Score"],
            np.nan
        )
        df_joined["Comb_relative"] = np.where(
            df_joined["comb_gt"].notnull() & (df_joined["comb_gt"]!=0),
            df_joined["Combined Score"] / df_joined["comb_gt"],
            np.nan
        )

        # 8) 输出
        out_file = f"../../../results/analysis_results/{ds}_summary_rel.xlsx"
        df_joined.to_excel(out_file, index=False)
        print(f"[INFO] => Wrote {out_file} with relative columns.")

if __name__=="__main__":
    main()
