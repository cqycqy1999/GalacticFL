from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'input': item['input'],
            'target': item['target']
        }

def get_data_loader(data_file, batch_size=32, shuffle=True):
    dataset = CustomDataset(data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)