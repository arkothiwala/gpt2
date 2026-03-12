import torch

class Attention(torch.nn.Module):
    def __init__(self, d_model, d_attention=None):
        self.d_model = d_model
        self.d_attention = d_attention
        if not d_attention:
            d_attention = d_model
        self.wq = torch.nn.Parameter(torch.empty(d_model, d_attention))
        self.wk = torch.nn.Parameter(torch.empty(d_model, d_attention))
        self.wv = torch.nn.Parameter(torch.empty(d_model, d_attention))

    def forward(self, x):
        x = x                       # (B, L, d_model)
        q = self.wq(x)              # (B, L_q, d_attention)
        k = self.wk(x)              # (B, L_k, d_attention)
        v = self.wv(x)              # (B, L_v, d_attention)
        k_t = k.permute(0,-1,-2)    # (B, d_attention, L_k)
        dot_product = q@k_t         # (B, L_q, L_k)
        scaled_dot_product = dot_product / (self.d_attention ** 0.5) # (B, L_q, L_k)
        attention = torch.nn.softmax(scaled_dot_product, dim=-1) # (B, L_q, L_k)
        return attention@v # (B, L_v, d_attention)