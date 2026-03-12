import torch
import numpy as np
class SinusoidalPositionalEmbeddings(torch.nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim

    def get_sine_wave(self, idx):
        return np.sin(idx/(10000**(np.arange(0,self.n_dim,2).reshape(1,-1)/self.n_dim)))

    def get_cos_wave(self, idx):
        return np.cos(idx/(10000**(np.arange(0,self.n_dim,2).reshape(1,-1)/self.n_dim)))

    # INITIAL FAULTY IMPLEMENTATION
    # def forward(self, idx):
    #     if isinstance(idx, int):
    #         print(idx)
    #         if idx%2 == 0:
    #             return self.get_sine_wave(idx)
    #         elif idx%2 == 1:
    #             return self.get_cos_wave(idx)
    #     else:
    #         sine_op = self.get_sine_wave(idx)*(1-np.arange(self.n_dim)%2)
    #         cos_op = self.get_cos_wave(idx)*(np.arange(self.n_dim)%2)
    #         return sine_op+cos_op

    def forward(self, seq_len):
        output = np.zeros(shape=(seq_len, self.n_dim))
        idx = np.arange(start=0, stop=seq_len, step=1).reshape(-1,1)
        output[:,::2] = self.get_sine_wave(idx)
        output[:,1::2] = self.get_cos_wave(idx)
        return output[:seq_len, :]