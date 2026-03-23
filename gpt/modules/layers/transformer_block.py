import torch
import collections
import numpy as np
from gpt.modules.embedding.sinusoidal import SinusoidalPositionalEmbeddings
from gpt.modules.norm.layernorm import CustomLayerNorm
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
        self.MHA = torch.nn.MultiheadAttention(
            embed_dim=self.d_model, 
            num_heads=self.n_heads,
            dropout=attention_dropout,
            bias=True,
            batch_first=True,
            # dtype=torch.bfloat16
        )
        self.FFN = torch.nn.Sequential(collections.OrderedDict([
            ("linear_expansion", torch.nn.Linear(in_features=d_model, out_features=4*d_model, bias=True)),
            # Mistake - I had initially forgotten the activation layer
            ("activation", torch.nn.GELU(approximate='tanh')),
            # ("dropout", torch.nn.Dropout(p=0.1)),
            ("linear_projection", torch.nn.Linear(in_features=4*d_model, out_features=d_model, bias=True))
        ]))
        
        self.dropout_residual_mha = torch.nn.Dropout(p=0.1)
        self.dropout_residual_ffn = torch.nn.Dropout(p=0.1)
        
        # self.logger.info(f"self.MHA.in_proj_weight.dtype = {self.MHA.in_proj_weight.dtype} | self.MHA.out_proj.weight.dtype = {self.MHA.out_proj.weight.dtype} | self.FFN.linear_projection.weight.dtype = {self.FFN.linear_projection.weight.dtype}")

        # scale MHA parameters by scaling factor
        # This was introduced in GPT2 paper to stabilize deep layers
        # ""We scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers.""

        ########################################################
        # I just realised that, this is wrong AS IT is scaling all the parameters including biases which should not be scaled.
        # Because, weights gets multiplied with prev. layer's output so they contribute in scaling variance but bias just get added so it doesn't contribute
        ########################################################
        # for parameter in self.MHA.parameters():
        #     parameter.data.mul_(scaling_factor)
        # for parameter in self.FFN.parameters():
        #     parameter.data.mul_(scaling_factor)

        # PARTIAL MISTAKE - I didn't do torch.no_grad() here that would lead to scaling factor being applied during backpropagation as well which is not what we want. We just want to scale the initial weights and not the gradients during backpropagation.
        # WHY PARTIAL MISTAKE? - I was directly updating `weight.data.mul_(scaling_factor)` which is an in-place operation and it would have been fine as we are not tracking gradients for weight.data but I wanted to be extra cautious and use torch.no_grad() to ensure that we are not tracking gradients for these operations. Also, using torch.no_grad() makes it clear to anyone reading the code that we are intentionally not tracking gradients for these operations.
        # with torch.no_grad():
            # scale only weights correctly
            # MISTAKE - Earlier I had scaled only in_proj_weight practically limiting the scenario where MHA is using separate q_proj_weight, k_proj_weight and v_proj_weight instead of combined in_proj_weight. This is because in some versions of PyTorch, MultiheadAttention uses separate projection weights for query, key and value instead of combined projection weight.
            # if self.MHA.in_proj_weight is not None:
            #     self.MHA.in_proj_weight.data.mul_(scaling_factor)
            # if self.MHA.q_proj_weight is not None:
            #     self.MHA.q_proj_weight.data.mul_(scaling_factor)
            # if self.MHA.k_proj_weight is not None:
            #     self.MHA.k_proj_weight.data.mul_(scaling_factor)
            # if self.MHA.v_proj_weight is not None:
            #     self.MHA.v_proj_weight.data.mul_(scaling_factor)
            # self.MHA.out_proj.weight.data.mul_(scaling_factor)
            # # self.FFN.linear_expansion.weight.data.mul_(scaling_factor)
            # self.FFN.linear_projection.weight.data.mul_(scaling_factor)
            # MISTAKE - I misunderstood the scaling factor and was scaling all the weights -> we need to just scale the residual connection weights which are out projection layer from MHA and FFN.

            # self.MHA.out_proj.weight.mul_(scaling_factor)
            # self.FFN.linear_projection.weight.mul_(scaling_factor)
            
            # # Added normal initialization for weights and zero initialization for biases as mentioned in GPT2 paper.
            # torch.nn.init.normal_(self.FFN.linear_expansion.weight, mean=0.0, std=0.02)
            # torch.nn.init.zeros_(self.FFN.linear_expansion.bias)
            # if self.MHA.in_proj_weight is not None:
            #     torch.nn.init.normal_(self.MHA.in_proj_weight, mean=0.0, std=0.02)
            #     torch.nn.init.zeros_(self.MHA.in_proj_bias)
            # if self.MHA.q_proj_weight is not None:
            #     torch.nn.init.normal_(self.MHA.q_proj_weight, mean=0.0, std=0.02)
            #     torch.nn.init.zeros_(self.MHA.q_proj_bias)
            # if self.MHA.k_proj_weight is not None:
            #     torch.nn.init.normal_(self.MHA.k_proj_weight, mean=0.0, std=0.02)
            #     torch.nn.init.zeros_(self.MHA.k_proj_bias)
            # if self.MHA.v_proj_weight is not None:
            #     torch.nn.init.normal_(self.MHA.v_proj_weight, mean=0.0, std=0.02)
            #     torch.nn.init.zeros_(self.MHA.v_proj_bias)
        
        self.layer_norm_mha = TorchLayerNorm(normalized_shape=self.d_model)
        self.layer_norm_ffn = TorchLayerNorm(normalized_shape=self.d_model)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(self.context_length, self.context_length)*float("-inf"), diagonal=1)
        )


    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_layer_norm_mha = self.layer_norm_mha(x)
        # self.logger.debug(f"x.shape = {x.shape} | x.device = {x.device} | x.dtype = {x.dtype}")
        # self.logger.debug(f"x_layer_norm_mha.shape = {x_layer_norm_mha.shape} | x_layer_norm_mha.device = {x_layer_norm_mha.device} | x_layer_norm_mha.dtype = {x_layer_norm_mha.dtype}")
        # MISTAKE - initially I had not provided separate query, key and values arguments. Also, I wasn't selecting first value
        # This was because in my custom implementation of MHA, I wasn't taking three inputs in MHA.
        x_post_mha, attention = self.MHA(
            query=x_layer_norm_mha, 
            key=x_layer_norm_mha, 
            value=x_layer_norm_mha, 
            # attn_mask=torch.triu(torch.ones(seq_len, seq_len)*float("-inf"), diagonal=1, device=x.device).bool(), # not using is_causal=True because because using it along with need_weights=True is leading to silent or loud failures based on the pytorch version. Also, instead of float mask we are preparing boolean mask as it would consume much less memory, makes compute slightly faster and is recommended by PyTorch.
            attn_mask=self.causal_mask[:seq_len, :seq_len], # using pre-registered buffer for causal mask. This would be more efficient as we are not creating the mask in every forward pass and also it would be on the same device as the model.
            need_weights=False,
            key_padding_mask=None, # given currently we are training models on full sequence length. This is additive mask. if key_padding_mask is boolean -> True is replaced with -inf and False is replaced with 0 in attention mask. if key_padding_mask is float -> values in key_padding_mask are directly added to attention mask. so we can directly provide key_padding_mask as attention mask for padding tokens.
            is_causal=True # using built-in causal mask support in PyTorch which is more efficient and also works well with need_weights=True. This would automatically apply the causal mask to the attention scores before softmax.
        )
        # self.logger.debug(f"x_post_mha.dtype = {x_post_mha.dtype}")
        x_post_mha = x + self.dropout_residual_mha(x_post_mha)
        # self.logger.info(f"self.MHA.in_proj_weight.dtype = {self.MHA.in_proj_weight.dtype} | self.MHA.out_proj.weight.dtype = {self.MHA.out_proj.weight.dtype} | self.FFN.linear_projection.weight.dtype = {self.FFN.linear_projection.weight.dtype}")

        x_layer_norm_ffn = self.layer_norm_ffn(x_post_mha)
        x_post_ffn = x_post_mha + self.dropout_residual_ffn(self.FFN(x_layer_norm_ffn))
        return x_post_ffn
        
