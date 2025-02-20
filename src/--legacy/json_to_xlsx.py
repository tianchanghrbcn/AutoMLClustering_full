import pandas as pd

# 读取 JSON 文件
df = pd.read_json("../../results/computed_results.json")

# 保存为 Excel
df.to_excel("../../results/computed_results.xlsx", index=False)
