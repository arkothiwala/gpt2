import torch

class RotaryPositionalEmbeddings(torch.nn.Module):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # (..., seq_len, n_dim)
        n_dim, seq_len = x.shape[-1], x.shape[-2]
        timestamps = torch.arange(start=0, end=seq_len, step=1)     # (seq_len,)
        theta = 10000**(-torch.arange(0,n_dim,2)/n_dim)             # (n_dim/2,)
        theta = theta.repeat_interleave(2, dim=-1)                          # (n_dim,)
        angle = torch.outer(timestamps,theta)                       # (seq_len,theta) == (m,theta)
        cos = angle.cos()                                           # (seq_len,theta) == (m,theta)
        sin = angle.sin()                                           # (seq_len,theta) == (m,theta)

        def rotate_half(x):
            output = torch.empty_like(x)
            x_odd, x_even = x[...,1::2], x[...,0::2]
            output[..., 1::2] = x_even
            output[..., 0::2] = -x_odd
            return output
        final = cos*x + sin*rotate_half(x)                          # (seq_len,theta) == (m,theta)
        return final