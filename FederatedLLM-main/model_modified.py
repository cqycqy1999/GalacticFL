import math
from typing import Optional # 从typing模块导入Optional类型，用于定义可选参数


# models to modified
from transformers import BloomPreTrainedModel, BloomModel, BloomConfig
# TODO 把Phi模型也导入

import torch.nn.functional as F
from torch import nn
import torch


class BloomAttentionWithSeparatedQKV(nn.Module):
    def __init__(self, config:BloomConfig, layer_idx: Optional[int] = None): # 传入BloomConfig对象，以及可选的层索引
        super().__init__() # 先初始化父类

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = nn.Dropout(config.attention_dropout) # 这里不是概率
        self.hidden_dropout = config.hidden_dropout # 这里只是概率

        if self.head_dim * self.num_heads != self.hidden_size:
                raise ValueError(
                    f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                    f" {self.num_heads})."
                )

        # 拆分QKV
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True) # bloom是有bias的，默认使用
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True) 

        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.pretraining_tp = config.pretraining_tp # 张量并行相关参数

        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0
        self.layer_idx = layer_idx
        if layer_idx is None:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

    
    def forward(self, 
                hidden_states: torch.Tensor,
                residual: torch.Tensor,
                alibi: torch.Tensor,
                attention_mask=None,
                ):
         # 生成Q、K、V（包含偏置项）
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # 调整形状，保证与原来的bloom形状一样
        batch_size, seq_len, _ = hidden_states.size()
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

