import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 定义所有的类别 
categories = [
    'open_qa', 'general_qa', 'classification',
    'closed_qa', 'brainstorming', 'information_extraction', 'summarization', 'creative_writing',
] 

# 初始化统计矩阵
client_stats = pd.DataFrame(0, index=range(10), columns=categories)

# 路径
data_path = "./data_Dolly/c10/alpha10" # TODO 更改

# 遍历所有客户端文件
for client_id in range(10):
    with open(os.path.join(data_path, "local_training_{}.json".format(client_id)), 'r') as f:
        data = json.load(f)

        # 统计类别分布
        df_client = pd.DataFrame(data)
        counts = df_client['category'].value_counts().reindex(categories, fill_value=0)

        # 记录到统计矩阵
        client_stats.loc[client_id] = counts

print("统计矩阵示例：")
print(client_stats.head())

# 将矩阵转换为长格式
bubble_data = client_stats.stack().reset_index()
bubble_data.columns = ['Client', 'Category', 'Count']

# 添加坐标映射
category_order = {cat: idx for idx, cat in enumerate(categories)}
bubble_data['y_coord'] = bubble_data['Category'].map(category_order)
bubble_data['x_coord'] = bubble_data['Client']

# 过滤零值（可选）
bubble_data = bubble_data[bubble_data['Count'] > 0]

print("\n气泡数据格式：")
print(bubble_data.head())

# 绘制基础气泡矩阵图
plt.figure(figsize=(16, 10))
ax = sns.scatterplot(
    data=bubble_data,
    x='x_coord',
    y='y_coord',
    size='Count',
    hue='Category',
    sizes=(50, 800),  # 气泡大小范围（最小/最大直径）
    alpha=0.7,
    palette='tab10',
    legend='brief'
)

# 坐标轴美化
plt.yticks(
    ticks=range(len(categories)),
    labels=categories,
    rotation=0,
    fontsize=12
)
plt.xticks(
    ticks=range(10),
    labels=[f'Client {i}' for i in range(10)],
    rotation=45,
    fontsize=12
)

# 网格线设置
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# 标签与标题
plt.xlabel('Client ID', fontsize=14, labelpad=15)
plt.ylabel('Category', fontsize=14, labelpad=15)
plt.title('Bubble Matrix: Category Distribution Across Clients_alpha10', fontsize=16, pad=20)

# 图例调整
plt.legend(
    bbox_to_anchor=(1.05, 1),
    borderpad=1,
    labelspacing=1.5,
    frameon=False
)

plt.tight_layout()
plt.savefig('debug10.png', dpi=120)
plt.show()