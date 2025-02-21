import pandas as pd

# 读取jsonl文件
df = pd.read_json("/home/archlab/cqy202/GalacticFL/FederatedLLM-main/data_Dolly/databricks-dolly-15k.jsonl", lines=True)

# 统计类别分布
category_dist = df['category'].value_counts()

print("完整类别分布：")
print(category_dist.to_markdown())

