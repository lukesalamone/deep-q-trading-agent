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
    def __init__(self, x:List):
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



def batched(dataset:str, batch_size=64) -> Dict[str, DataLoader]:
    """
    Each batch sample contains (state, next_state, price, prev_price) where
        state       p_t - p_{t-1} for t-199 to t where p_t is the closing price
                    on day t
        next_state  identical to state but shifted forward by one day
                    e.g. next_state[0] == state[1]
        price       closing price for day t
        prev_price  closing price for day t-1

    :param dataset: which stock or index history to be loaded
    :param batch_size: batch size
    :return:
    """
    batches = {}
    datasets = _load_from_file(dataset)

    for key in datasets:
        batch = []
        prices = datasets[key]

        a = torch.Tensor(prices[1:])
        b = torch.Tensor(prices[:-1])
        states = a - b

        for i in range(len(states) - LOOKBACK):
            price = prices[LOOKBACK+i-1]
            prev_price = prices[LOOKBACK+i-2]
            state = states[i:LOOKBACK+i]
            next_state = states[i+1:LOOKBACK+i+1]
            sample = (state, next_state, price, prev_price)
            batch.append(sample)

        ds = Sequence_Dataset(x=batch)
        batches[key] = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)

    return batches


if __name__ == '__main__':
    # batches = batched('sx5e')

    prices = [3,1,4,1,5,9,2,6,5,3,5,8,9]
    batch = []
    LOOKBACK = 5

    a = torch.Tensor(prices[1:])
    b = torch.Tensor(prices[:-1])
    states = a - b

    for i in range(len(states) - LOOKBACK):
        price = prices[LOOKBACK + i - 1]
        prev_price = prices[LOOKBACK + i - 2]
        state = states[i:LOOKBACK + i]
        next_state = states[i + 1:LOOKBACK + i + 1]
        sample = (state, next_state, price, prev_price)
        batch.append(sample)

    print(batch)
    