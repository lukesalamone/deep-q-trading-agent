import os
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader


ALLOWED_DATASETS = {'gspc', 'hsi', 'ks11', 'sx5e'}
INPUT_PATH = 'data_clean'
LOOKBACK = 200

class Sequence_Dataset(Dataset):
    """
    Container to hold state values.
    Each state s_t contains the
    """
    def __init__(self, x:torch.LongTensor):
        self.x = x
        self.len = len(x)

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return self.len

def _load_from_file(dataset:str) -> Dict[str, List[float]]:
    dataset = dataset.lower()
    if dataset not in ALLOWED_DATASETS:
        raise ValueError(f'Dataset type {dataset} not allowed')
    else:
        files = {}
        for file in os.listdir(INPUT_PATH):
            if not file.startswith(dataset):
                continue

            ds = file.split('.')[1]

            with open(os.path.join(INPUT_PATH, file), 'r') as f:
                files[ds] = [float(x) for x in f.read().split(',')]

        return files


# read time series data in
# each sample is
#     state s = p_{t} - p_{t-1} for 200 previous days
def batched(dataset:str, batch_size=64) -> Dict[str, DataLoader]:
    batches = {}
    datasets = _load_from_file(dataset)

    for key in datasets:
        batch = []
        series = datasets[key]
        for i in range(len(series)):
            if i-LOOKBACK-1 < 0:
                continue

            sample_today = torch.Tensor(series[i-LOOKBACK:i])
            sample_yesterday = torch.Tensor(series[i-LOOKBACK-1:i-1])
            sample = sample_today - sample_yesterday
            batch.append(sample)

        ds = Sequence_Dataset(x=batch)
        batches[key] = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)

    return batches


if __name__ == '__main__':
    batches = batched('sx5e')
    print(batches)
    