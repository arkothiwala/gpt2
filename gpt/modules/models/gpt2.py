import torch
import numpy as np
from gpt.modules.layers.transformer_block import CustomLayerNorm, TransformerBlock

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
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        # self.position_embedding = SinusoidalPositionalEmbeddings(n_dim=self.d_model)
        # add sequential layers
        for layer in range(self.n_layers):
            self.transformer_layers.append(
                TransformerBlock(
                    d_model=self.d_model, 
                    n_heads=self.n_heads, 
                    scaling_factor=1/np.sqrt(2*self.n_layers), 
                    context_length=self.context_length,
                    attention_dropout=0.1
                )
            )
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
        torch.nn.init.normal_(self.learnt_position_embedding.weight, mean=0.0, std=0.02)
        self.dropout = torch.nn.Dropout(p=0.1)


    def forward(self, x, return_proba = False):
        # print(f"x.shape = {x.shape}")
        batch_size, seq_len = x.shape
        x_learnt_embeddings = self.embedding(x)
        # x_pos_embeddings = self.position_embedding(self.context_length)
        x_pos_embeddings = self.learnt_position_embedding(torch.arange(start=0, end=seq_len).to(x.device))
        x_embeddings = x_learnt_embeddings + x_pos_embeddings
        # x_embeddings = torch.nn.functional.dropout(input=x_embeddings, p=0.1) # MISTAKE - I had initially used functional dropout here w/o train v/s inference mode check. Moving it to Dropout module which internally manages train v/s inference mode and also makes code cleaner.
        x_embeddings = self.dropout(x_embeddings)
        x_logits = self.transformer_layers(x_embeddings)
        x_logits = x_logits@self.embedding.weight.T
        if return_proba:
            return torch.nn.functional.softmax(input=x_logits, dim=-1)
        else:
            return x_logits