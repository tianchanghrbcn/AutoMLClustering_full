import pandas as pd

# 读取 JSON 文件
df = pd.read_json("../../results/computed_results.json")

# 保存为 CSV
df.to_csv("../../results/computed_results.csv", index=False)
