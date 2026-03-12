1. Dataset creation:
    - Implemented random length and random start for training data. Later switched to random length, fixed start [thinking model may not learn well if it starts at a random location]
    - We can use tiktokenizer encode_batch with optimal no of threads to reduce the encode time [did benchmarking as well - benchmarking/tokeniser_batch_encode.py]
1. Implemented masking when I shouldn't have
    - Implemeted dynamic length rnn masking [similar to done in timeseries V1 and V2], realised that this may not be correct. Checked whether one should use fixed length masking instead of dynamic given model context length is constant
    - Realised that for LLM pretraining we do fixed length **sequence packing**

    - I was applying masking to set prediction values to -100 where x was endoftext token => realised that I had applied EOT mask on list instead of tensor hence my code was breaking misrably


MHA:
- I was initally thinking k_dim and v_dim as the dimensions on which we want embedding dimensions to project. It wasn't true. It is the dimensions in which one can expect K and V tensors to originally have shape. MHA internally projects them to match d_model dimensions
- I had applied scaling for MHA weights by torch.sqrt(1/self.n_layers) where as it should have been torch.sqrt(1/(2*self.n_layers)) because the network will have 2*self.n_layers number of residual blocks. Each block is increasing variance hence we need to normalise accordingly. Another way to think is FFN weight normalization is taking care of two residual connections - one before and one after the FFN