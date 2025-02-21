import sys
import pandas as pd
import numpy as np
import random
import os
import json
import pdb

import abc
import random
import inspect

# num_clients = int(sys.argv[1]) # 执行该py文件传入的第一个参数 int
# diff_quantity = int(sys.argv[2]) # 执行该py文件传入的第二个参数 bool

np.random.seed(42)
random.seed(42)

class BaseSplitter(abc.ABC):
    """
    This is an abstract base class for all splitter, which is not \
    implemented with ``__call__()``.

    Attributes:
        client_num: Divide the dataset into ``client_num`` pieces.
    """
    def __init__(self, client_num):
        self.client_num = client_num

    @abc.abstractmethod
    def __call__(self, dataset, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        """

        Returns: Meta information for `Splitter`.

        """
        sign = inspect.signature(self.__init__).parameters.values()
        meta_info = tuple([(val.name, getattr(self, val.name))
                           for val in sign])
        return f'{self.__class__.__name__}{meta_info}'

# Divide the entire dataset into a training set and a test set.

def dirichlet_distribution_noniid_slice(label,
                                        client_num,
                                        alpha,
                                        min_size=1,
                                        prior=None):
    r"""Get sample index list for each client from the Dirichlet distribution.
    https://github.com/FedML-AI/FedML/blob/master/fedml_core/non_iid
    partition/noniid_partition.py

    Arguments:
        label (np.array): Label list to be split.
        client_num (int): Split label into client_num parts.
        alpha (float): alpha of LDA.
        min_size (int): min number of sample in each client
    Returns:
        idx_slice (List): List of splited label index slice.
    """
    if len(label.shape) != 1:
        raise ValueError('Only support single-label tasks!')

    if prior is not None:
        return _split_according_to_prior(label, client_num, prior)

    num = len(label)
    classes = len(np.unique(label))
    assert num > client_num * min_size, f'The number of sample should be ' \
                                        f'greater than' \
                                        f' {client_num * min_size}.'
    size = 0
    while size < min_size:
        idx_slice = [[] for _ in range(client_num)]
        for k in range(classes):
            # for label k
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(alpha, client_num))
            # prop = np.array([
            #    p * (len(idx_j) < num / client_num)
            #    for p, idx_j in zip(prop, idx_slice)
            # ])
            # prop = prop / sum(prop)
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            idx_slice = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))
            ]
            size = min([len(idx_j) for idx_j in idx_slice])
    for i in range(client_num):
        np.random.shuffle(idx_slice[i])
    return idx_slice


class IIDSplitter(BaseSplitter):
    """
    This splitter splits dataset following the independent and identically \
    distribution.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
    """
    def __init__(self, client_num):
        super(IIDSplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None):
        from torch.utils.data import Dataset, Subset

        length = len(dataset)
        index = [x for x in range(length)]
        np.random.shuffle(index)
        idx_slice = np.array_split(np.array(index), self.client_num)
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list


class LDASplitter(BaseSplitter):
    """
    This splitter split dataset with LDA.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
        alpha (float): Partition hyperparameter in LDA, smaller alpha \
            generates more extreme heterogeneous scenario see \
            ``np.random.dirichlet``
    """
    def __init__(self, client_num, alpha=0.5):
        self.alpha = alpha
        super(LDASplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        tmp_dataset = [ds for ds in dataset]
        if isinstance(tmp_dataset[0], tuple):
            label = np.array([y for x, y in tmp_dataset])
        elif isinstance(tmp_dataset[0], dict):
            label = np.array([x['categories'] for x in tmp_dataset])
        else:
            raise TypeError(f'Unsupported data formats {type(tmp_dataset[0])}')
        idx_slice = dirichlet_distribution_noniid_slice(label,
                                                        self.client_num,
                                                        self.alpha,
                                                        prior=prior)
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list


class MetaSplitter(BaseSplitter):   
    def __init__(self, client_num, **kwargs):
        super(MetaSplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        tmp_dataset = [ds for ds in dataset]
        if isinstance(tmp_dataset[0], tuple):
            label = np.array([y for x, y in tmp_dataset])
        elif isinstance(tmp_dataset[0], dict):
            label = np.array([x['categories'] for x in tmp_dataset])
        else:
            raise TypeError(f'Unsupported data formats {type(tmp_dataset[0])}')

        # Split by categories
        categories = set(label)
        idx_slice = []
        for cat in categories:
            idx_slice.append(np.where(np.array(label) == cat)[0].tolist())
        random.shuffle(idx_slice)

        # Merge to client_num pieces
        new_idx_slice = []
        for i in range(len(categories)):
            if i < self.client_num:
                new_idx_slice.append(idx_slice[i])
            else:
                new_idx_slice[i % self.client_num] += idx_slice[i]

        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list


def get_splitter(num_clients, split_type, **kwargs):
    """
    Get the splitter by the given type.

    Args:
        num_clients: the number of clients
        split_type: the type of splitter
        **kwargs: the arguments of the splitter

    Returns:
        The splitter object
    """
    if split_type == 'iid':
        splitter = IIDSplitter(num_clients)
    elif split_type == 'lda':
        splitter = LDASplitter(num_clients, **kwargs)
    elif split_type == 'meta':
        splitter = MetaSplitter(num_clients, **kwargs)
    else:
        raise ValueError(f"Unsupported split type: {split_type}")
    
    return splitter


if __name__ == "__main__":
    num_clients = 10
    diff_quantity = True

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
