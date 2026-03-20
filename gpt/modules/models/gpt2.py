import torch
import numpy as np
from gpt.modules.layers.transformer_block import CustomLayerNorm, TransformerBlock
import logging
logger = logging.getLogger(__name__)

class GPT2Model(torch.nn.Module):
    def __init__(self, d_model, n_heads, n_layers, vocab_size, context_length, logger=logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.transformer_layers = torch.nn.Sequential()
        self.final_layer_norm = CustomLayerNorm(d_model=self.d_model)
        self.logger = logger
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
            padding_idx=None,
            max_norm=None, # changing max_norm to None -> will let model figure unless the training is unstable and we see exploding gradients.
            norm_type=2
        )
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
        self.dropout = torch.nn.Dropout(p=0.1)
        self.initialize_model_parameters(scaling_factor=1/np.sqrt(2*self.n_layers))
        self.print_weight_std(self.transformer_layers)

    def print_weight_std(self, model):
        self.logger.info(f"{'Layer Name':<60} | {'Std Dev':<10}")
        self.logger.info("-" * 75)
        
        for name, param in model.named_parameters():
            # We only care about weights, not biases
            if 'weight' in name:
                std = param.std().item()
                self.logger.info(f"{name:<60} | {std:.6f}")
        
    def initialize_model_parameters(self, scaling_factor):
        self.logger.info(f"scaling_factor = {scaling_factor}")
        with torch.no_grad():
            # initialize all linear weights with normal distribution and biases with zeros
            for module in self.modules():
                self.logger.info(f"Initializing module: {module.__class__.__name__}")
                if isinstance(module, torch.nn.Linear):
                    self.logger.info(f"setting {module._get_name()} weight with normal distribution and bias with zeros")
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                if isinstance(module, torch.nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    
                if isinstance(module, CustomLayerNorm):
                    self.logger.info("Initializing CustomLayerNorm weight with ones and bias with zeros")
                    # print weights and bias norms for confirmation
                    # self.logger.info(f"Before initialization - LayerNorm weight norm: {module.weight.norm().item()}, bias norm: {module.bias.norm().item()}")
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.bias)
                    # self.logger.info(f"After initialization - LayerNorm weight norm: {module.weight.norm().item()}, bias norm: {module.bias.norm().item()}")
                    
                if isinstance(module, TransformerBlock):
                    # scale only residual connection weights correctly
                    torch.nn.init.normal_(module.MHA.in_proj_weight, mean=0.0, std=0.02)
                    torch.nn.init.normal_(module.MHA.out_proj.weight, mean=0.0, std=0.02)
                    self.logger.info("Initializing TransformerBlock MHA in_proj_weight with normal distribution")
                    
                    module.MHA.out_proj.weight.mul_(scaling_factor)
                    module.FFN.linear_projection.weight.mul_(scaling_factor)


    def forward(self, x, return_proba = False):
        # self.logger.info(f"x.shape = {x.shape}")
        batch_size, seq_len = x.shape
        assert seq_len <= self.context_length, "Sequence length exceeds model context_length"
        x_learnt_embeddings = self.embedding(x)
        # x_pos_embeddings = self.position_embedding(self.context_length)
        x_pos_embeddings = self.learnt_position_embedding(torch.arange(start=0, end=seq_len, device=x.device)).unsqueeze(0)
        
        x_embeddings = x_learnt_embeddings + x_pos_embeddings
        # x_embeddings = torch.nn.functional.dropout(input=x_embeddings, p=0.1) # MISTAKE - I had initially used functional dropout here w/o train v/s inference mode check. Moving it to Dropout module which internally manages train v/s inference mode and also makes code cleaner.
        x_embeddings = self.dropout(x_embeddings)
        x_logits = self.transformer_layers(x_embeddings)
        x_logits = x_logits@self.embedding.weight.T
        if return_proba:
            return torch.nn.functional.softmax(input=x_logits, dim=-1)
        else:
            return x_logits