# models to modified
from transformers import BloomModel, BloomConfig

from torch import nn
import torch

class BloomAttentionWithSeparatedQKV(nn.Module):
    def __init__(self, config:BloomConfig):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
