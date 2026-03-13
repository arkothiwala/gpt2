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
- Silly mistake -> I had not set `batch_first = True` explicitly and torch set `batch_first = False` by default
- In forward pass, I didn't pass x for all query, key and value arguments -> apperently they are different due to common API for MHA
- had not applied causal_mask which I realised later when I was preparing for the training.
- MHA returns two values -> post MHA output, and actual attention
- Identified a bug in torch -> if is_causal=True and need_weights=True [by default] => it would not apply causal mask and fail silently
    - option 1 - either set need_weights=False
    - [used this] option 2 - create causal mask and pass it in the attn_mask
- was doing `if self.MHA.in_proj_weight:` this throws error as boolean comparison goes for a toss when the tensor has a value -> should use `self.MHA.in_proj_weight is not None` instead because torch has a native support to check None
- I misunderstood the scaling factor for residual layers and was scaling all the weights -> we need to just scale the residual connection weights which are out projection layer from MHA and FFN.
- while applying scaling I did `self.MHA.out_proj.weight.data.mul_(scaling_factor)` -> doing weight.data.mul_ bypasses torch's autograd engine. this is discouraged in torch

Position Encoding:
- was returning numpy array instead of torch.tensor -> leading to issue when doing sum b/w tensor and ndarray
- was passing input X instead of input X's seq_len
- when creating torch.tensor() had to set dtype to float32 because torch by default set dtype to float32 v/s numpy sets it to float64 or double which makes operations incompatible
- was using fixed sinusoidal embeddings [as in attention is all you need] -> GPT2 is using learnd embeedings instead
- use register_buffer - I am calculating torch.triu for each forward pass for each layer -> this can make it super slow -> better to register at initialization and use it as required by the current seq_len

Embeddings:
- max_norm can be left None unless the training is unstable
- was creating another layer in the last layer to translate d_model to vocab_size. This will blow up the parameter space. GPT1 and GPT2 uses the same embedding metrics transpose for it.

Dropout:
- I was applying dropout w/o training v/s inference flag. This would have created dropout even in the inference mode [suboptimal]
- **NN.module manage bunch of such things which torch.nn.functional does not. ALWAYS be SUPER careful while using torch.nn.functional over equivalent torch.nn.Module**