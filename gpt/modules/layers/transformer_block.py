import torch
import collections


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
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x-mean)/torch.sqrt(var + self.eps)
        x_normalized = (x_normalized*self.weight)+self.bias
        return x_normalized

class TransformerBlock(torch.nn.Module):

    def __init__(self, d_model, n_heads, attention_dropout=0.2, scaling_factor=1, *args, **kwargs):
        self.d_model = d_model
        self.n_heads = n_heads
        self.MHA = torch.nn.MultiheadAttention(
            embed_dim=self.d_model, 
            num_heads=self.n_heads,
            dropout=attention_dropout,
            bias=True
        )
        self.FFN = torch.nn.Sequential(collections.OrderedDict([
            ("linear_expansion", torch.nn.Linear(in_features=d_model, out_features=4*d_model, bias=True)),
            # Mistake - I had initially forgotten the activation layer
            ("activation", torch.nn.GELU()),
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
        self.MHA.in_proj_weight.data.mul_(scaling_factor)
        self.MHA.out_proj.weight.data.mul_(scaling_factor)
        self.FFN.linear_expansion.weight.data.mul_(scaling_factor)
        self.FFN.linear_projection.weight.data.mul_(scaling_factor)
        
        self.layer_norm_mha = CustomLayerNorm(d_model=self.d_model)
        self.layer_norm_ffn = CustomLayerNorm(d_model=self.d_model)

    def forward(self, x):
        x_layer_norm_mha = self.layer_norm_mha(x)
        x_post_mha = x + self.MHA(x_layer_norm_mha)

        x_layer_norm_ffn = self.layer_norm_ffn(x_post_mha)
        x_post_ffn = x_post_mha + self.FFN(x_layer_norm_ffn)
        return x_post_ffn