# import torch
# import numpy as np
# from functools import lru_cache
# class SinusoidalPositionalEmbeddings(torch.nn.Module):
#     def __init__(self, n_dim):
#         super().__init__()
#         self.n_dim = n_dim

#     def get_sine_wave(self, idx):
#         return np.sin(idx/(10000**(np.arange(0,self.n_dim,2).reshape(1,-1)/self.n_dim)))

#     def get_cos_wave(self, idx):
#         return np.cos(idx/(10000**(np.arange(0,self.n_dim,2).reshape(1,-1)/self.n_dim)))

#     # INITIAL FAULTY IMPLEMENTATION
#     # def forward(self, idx):
#     #     if isinstance(idx, int):
#     #         print(idx)
#     #         if idx%2 == 0:
#     #             return self.get_sine_wave(idx)
#     #         elif idx%2 == 1:
#     #             return self.get_cos_wave(idx)
#     #     else:
#     #         sine_op = self.get_sine_wave(idx)*(1-np.arange(self.n_dim)%2)
#     #         cos_op = self.get_cos_wave(idx)*(np.arange(self.n_dim)%2)
#     #         return sine_op+cos_op

#     @lru_cache(maxsize=None)
#     def forward(self, seq_len):
#         output = np.zeros(shape=(seq_len, self.n_dim))
#         idx = np.arange(start=0, stop=seq_len, step=1).reshape(-1,1)
#         output[:,::2] = self.get_sine_wave(idx)
#         output[:,1::2] = self.get_cos_wave(idx)
#         return torch.tensor(output[:seq_len, :], dtype=torch.float32)

import torch
import math

class SinusoidalPositionalEmbeddings(torch.nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create a matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        
        # Create a column vector of positions: [[0], [1], [2], ...]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the scaling factor in log space for numerical stability
        # Equivalent to: 1 / (10000 ** (2i / d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply Sine to even indices: 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply Cosine to odd indices: 2i + 1
        # Handle the edge case where d_model is odd
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        # Register as a buffer: shape (1, max_seq_len, d_model) for easy broadcasting
        # It won't be updated by the optimizer, but will move to the correct device
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Positional embeddings of shape (1, seq_len, d_model)
        """
        seq_len = x.size(1)
        # Simply slice the pre-computed buffer up to the current sequence length
        return self.pe[:, :seq_len, :]