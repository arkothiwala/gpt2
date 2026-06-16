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

class CustomMultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=None, bias=True, batch_first=True):
        super().__init__()
        self.d_model = embed_dim
        self.n_heads = num_heads
        self.dropout = dropout if dropout else 0.0
        self.d_attention = self.d_model//self.n_heads
        self.in_proj = torch.nn.Linear(in_features=self.d_model, out_features=3*self.d_model, bias=bias)
        self.in_proj_weight = self.in_proj.weight
        self.out_proj = torch.nn.Linear(in_features=self.d_attention*self.n_heads, out_features=self.d_model, bias=bias)
        if batch_first==False:
            raise ValueError("this implementation only supports batch_first at the moment.")

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        x = self.in_proj(x) # (batch_size, seq_len, 3*d_model)
        x = x.reshape(batch_size, seq_len, 3, self.n_heads, self.d_attention).permute(2, 0, 3, 1, 4).contiguous()
        q,k,v = torch.unbind(x, dim=0)
        # print(f"q.is_contiguous() = {q.is_contiguous()} | q.dtype = {q.dtype} | q.shape={q.shape}")
        # print(f"k.is_contiguous() = {k.is_contiguous()} | k.dtype = {k.dtype} | k.shape={k.shape}")
        # print(f"v.is_contiguous() = {v.is_contiguous()} | v.dtype = {v.dtype} | v.shape={v.shape}")
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
        out = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=True
        ).permute(
            0,2,1,3
        ).contiguous().reshape(
            batch_size, seq_len, d_model
        )
        out = self.out_proj(out)
        return out