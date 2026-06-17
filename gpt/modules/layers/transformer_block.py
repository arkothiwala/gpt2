import torch
import collections
import numpy as np
from gpt.modules.embedding.sinusoidal import SinusoidalPositionalEmbeddings
from gpt.modules.norm.layernorm import CustomLayerNorm
from gpt.modules.layers.attention import CustomMultiHeadAttention
from torch.nn import LayerNorm as TorchLayerNorm

import logging
logger = logging.getLogger(__name__)
class TransformerBlock(torch.nn.Module):

    def __init__(self, d_model, n_heads, context_length, attention_dropout=0.1, scaling_factor=1, logger=logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.context_length = context_length
        self.logger = logger
        self.layer_norm_mha = TorchLayerNorm(normalized_shape=self.d_model)
        # self.MHA = torch.nn.MultiheadAttention(
        self.MHA = CustomMultiHeadAttention(
            embed_dim=self.d_model, 
            num_heads=self.n_heads,
            dropout=attention_dropout,
            bias=True,
            batch_first=True,
            # dtype=torch.bfloat16
        )
        self.layer_norm_ffn = TorchLayerNorm(normalized_shape=self.d_model)
        self.FFN = torch.nn.Sequential(collections.OrderedDict([
            ("linear_expansion", torch.nn.Linear(in_features=d_model, out_features=4*d_model, bias=True)),
            # Mistake - I had initially forgotten the activation layer
            ("activation", torch.nn.GELU(approximate='tanh')),
            ("dropout", torch.nn.Dropout(p=0.1)),
            ("linear_projection", torch.nn.Linear(in_features=4*d_model, out_features=d_model, bias=True))
        ]))
        
        self.dropout_residual_mha = torch.nn.Dropout(p=0.1)
        self.dropout_residual_ffn = torch.nn.Dropout(p=0.1)


    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        # x_post_mha = self.layer_norm_mha(x + self.dropout_residual_mha(self.MHA(x=x)))
        x_post_mha = x + self.dropout_residual_mha(self.MHA(x=self.layer_norm_mha(x)))
        x_post_ffn = x_post_mha + self.dropout_residual_ffn(self.FFN(self.layer_norm_ffn(x_post_mha)))
        return x_post_ffn
        
