import time
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

class NumQModel(nn.Module):
    def __init__(self):
        super().__init__()

