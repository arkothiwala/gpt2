import torch
import collections
import numpy as np
from gpt.modules.embedding.sinusoidal import SinusoidalPositionalEmbeddings

class CustomLayerNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.weight = torch.nn.Parameter(data=torch.ones(size=(self.d_model,)))
        self.bias = torch.nn.Parameter(data=torch.zeros(size=(self.d_model,)))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False because we want to divide by N instead of N-1, because we are calculating the variance for the entire population (the sequence) and not a sample of the population.
        # torch by default uses unbiased=True, which divides by N-1, so we need to set it to False to divide by N. Leading to incorrect results and mismatch with torch.nn.LayerNorm results.
        # MISTAKE - initially I had not set it to false there for results weren't matching with torch.nn.LayerNorm
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x-mean)/torch.sqrt(var + self.eps)
        x_normalized = (x_normalized*self.weight)+self.bias
        return x_normalized

class TransformerBlock(torch.nn.Module):

    def __init__(self, d_model, n_heads, attention_dropout=0.1, scaling_factor=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.MHA = torch.nn.MultiheadAttention(
            embed_dim=self.d_model, 
            num_heads=self.n_heads,
            dropout=attention_dropout,
            bias=True,
            batch_first=True
        )
        self.FFN = torch.nn.Sequential(collections.OrderedDict([
            ("linear_expansion", torch.nn.Linear(in_features=d_model, out_features=4*d_model, bias=True)),
            # Mistake - I had initially forgotten the activation layer
            ("activation", torch.nn.GELU()),
            ("dropout", torch.nn.Dropout(p=0.1)),
            ("linear_projection", torch.nn.Linear(in_features=4*d_model, out_features=d_model, bias=True))
        ]))

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
        self.MHA.out_proj.weight.data.mul_(scaling_factor)
        # self.FFN.linear_expansion.weight.data.mul_(scaling_factor)
        self.FFN.linear_projection.weight.data.mul_(scaling_factor)
        # MISTAKE - I misunderstood the scaling factor and was scaling all the weights -> we need to just scale the residual connection weights which are out projection layer from MHA and FFN.
        
        self.layer_norm_mha = CustomLayerNorm(d_model=self.d_model)
        self.layer_norm_ffn = CustomLayerNorm(d_model=self.d_model)


    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_layer_norm_mha = self.layer_norm_mha(x)
        # MISTAKE - initially I had not provided separate query, key and values arguments. Also, I wasn't selecting first value
        # This was because in my custom implementation of MHA, I wasn't taking three inputs in MHA.
        x_post_mha, attention = self.MHA(
            query=x_layer_norm_mha, 
            key=x_layer_norm_mha, 
            value=x_layer_norm_mha, 
            attn_mask=torch.triu(torch.ones(seq_len, seq_len)*float("-inf"), diagonal=1).bool(), # not using is_causal=True because because using it along with need_weights=True is leading to silent or loud failures based on the pytorch version. Also, instead of float mask we are preparing boolean mask as it would consume much less memory, makes compute slightly faster and is recommended by PyTorch.
            need_weights=True,
            key_padding_mask=None # given currently we are training models on full sequence length. This is additive mask. if key_padding_mask is boolean -> True is replaced with -inf and False is replaced with 0 in attention mask. if key_padding_mask is float -> values in key_padding_mask are directly added to attention mask. so we can directly provide key_padding_mask as attention mask for padding tokens.
        )
        x_post_mha = x + x_post_mha
        

        x_layer_norm_ffn = self.layer_norm_ffn(x_post_mha)
        x_post_ffn = x_post_mha + self.FFN(x_layer_norm_ffn)
        return x_post_ffn
        

class GPT2Model(torch.nn.Module):
    def __init__(self, d_model, n_heads, n_layers, vocab_size, context_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.transformer_layers = torch.nn.Sequential()
        self.final_layer_norm = CustomLayerNorm(d_model=self.d_model)
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
            padding_idx=None,
            max_norm=None, # changing max_norm to None -> will let model figure unless the training is unstable and we see exploding gradients.
            norm_type=2
        )
        self.position_embedding = SinusoidalPositionalEmbeddings(n_dim=self.d_model)
        # add sequential layers
        for layer in range(self.n_layers):
            self.transformer_layers.append(TransformerBlock(d_model=self.d_model, n_heads=self.n_heads, scaling_factor=1/np.sqrt(2*self.n_layers)), attention_dropout=0.1)
        # add final layer normalization
        self.transformer_layers.append(self.final_layer_norm)
        # predict token with softmax
        # self.transformer_layers.append(torch.nn.Linear(in_features=self.d_model, out_features=self.vocab_size))
        self.learnt_position_embedding = torch.nn.Embedding(
            num_embeddings=self.context_length,
            embedding_dim=self.d_model,
            padding_idx=None,
            max_norm=None, # changing max_norm to None -> will let model figure unless the training is unstable and we see exploding gradients.
            norm_type=2
        )


    def forward(self, x, return_proba = False):
        x_learnt_embeddings = self.embedding(x)
        # x_pos_embeddings = self.position_embedding(self.context_length)
        x_pos_embeddings = self.learnt_position_embedding(torch.arange(start=0, end=self.context_length).to(x.device))
        x_embeddings = x_learnt_embeddings + x_pos_embeddings
        x_embeddings = torch.nn.functional.dropout(input=x_embeddings, p=0.1)
        x_logits = self.transformer_layers(x_embeddings)
        x_logits = x_logits@self.embedding.weight.T
        if return_proba:
            return torch.nn.functional.softmax(input=x_logits, dim=-1)
        else:
            return x_logits