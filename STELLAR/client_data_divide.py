import os
import sys
import json
import random

import numpy as np
import pandas as pd

num_clients = int(sys.argv[1])
diff_quantity = int(sys.argv[2])

np.random.seed(42)
random.seed(42)

# 把数据集分为训练集和测试集
df = pd.read_json("./data_Dolly/databricks-dolly-15k.jsonl", lines=True) # json文件是orient='records'；jsonl需要修改
sorted_df = df.sort_values(by=['category'])
grouped = sorted_df.groupby('category')
sampled_df = grouped.apply(lambda x: x.sample(n=10)) # 每一类抽取 10 条数据作为test数据
sampled_df = sampled_df.reset_index(level=0, drop=True)
remaining_df = sorted_df.drop(index=sampled_df.index)

sampled_df = sampled_df.reset_index().drop('index', axis=1)
remaining_df = remaining_df.reset_index().drop('index', axis=1)
data_path = "./data_Dolly/c10/alpha10" # TODO 更改

# absolute_path = os.path.abspath(data_path)
# print("绝对路径:", absolute_path)

os.makedirs(data_path,exist_ok=True)

remaining_df_dic = remaining_df.to_dict(orient='records')
print('*'*5, len(remaining_df_dic))
with open(os.path.join(data_path, "global_training.json"), 'w') as outfile:
    json.dump(remaining_df_dic, outfile)

sampled_df_dic = sampled_df.to_dict(orient='records')
print('-'*5, len(sampled_df_dic))
with open(os.path.join(data_path, "global_test.json"), 'w') as outfile:
    json.dump(sampled_df_dic, outfile)

# Partition the global training data into smaller subsets for each client's local training dataset

if diff_quantity: # TODO Non-IID 
    min_size = 0 # 最短的数据
    min_require_size = 40 # 最长的数据
    alpha = 1.0 # 异构超参数

    N = len(remaining_df)
    net_dataidx_map = {}
    category_uniques = remaining_df['category'].unique().tolist()
    while min_size < min_require_size:

        idx_partition = [[] for _ in range(num_clients)]

        # print('*'*20)
        # print(type(idx_partition))
        # print(idx_partition)
        # print('*'*20)

        for k in range(len(category_uniques)):
            category_rows_k = remaining_df.loc[remaining_df['category'] == category_uniques[k]]
            category_rows_k_index = category_rows_k.index.values
            np.random.shuffle(category_rows_k_index)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_partition)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(category_rows_k_index)).astype(int)[:-1]
            idx_partition = [idx_j + idx.tolist() for idx_j, idx in
                            zip(idx_partition, np.split(category_rows_k_index, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_partition])

            # print('-'*20)
            # print(type(idx_partition))
            # print(idx_partition)
            # print('-'*20)

        print(min_size)


else: # TODO IID
    num_shards_per_clients = 2
    remaining_df_index = remaining_df.index.values
    shards = np.array_split(remaining_df_index, int(num_shards_per_clients * num_clients))
    random.shuffle(shards)

    shards = [shards[i:i + num_shards_per_clients] for i in range(0, len(shards), num_shards_per_clients)]
    idx_partition = [np.concatenate(shards[n]).tolist() for n in range(num_clients)]


for client_id, idx in enumerate(idx_partition):
    print(
        "\n Generating the local training dataset of Client_{}".format(client_id)
    )
    sub_remaining_df = remaining_df.loc[idx]
    sub_remaining_df = sub_remaining_df.reset_index().drop('index', axis=1)
    sub_remaining_df_dic = sub_remaining_df.to_dict(orient='records')

    with open(os.path.join(data_path, "local_training_{}.json".format(client_id)), 'w') as outfile:
        json.dump(sub_remaining_df_dic, outfile)
