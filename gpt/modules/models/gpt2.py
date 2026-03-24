import torch
import numpy as np
from gpt.modules.layers.transformer_block import CustomLayerNorm, TransformerBlock
from torch.nn import LayerNorm as TorchLayerNorm
from gpt.modules.embedding.sinusoidal import SinusoidalPositionalEmbeddings
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
        self.logger = logger
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
            padding_idx=None,
            max_norm=None, # changing max_norm to None -> will let model figure unless the training is unstable and we see exploding gradients.
            norm_type=2
        )
        # self.learnt_position_embedding = torch.nn.Embedding(
        #     num_embeddings=self.context_length,
        #     embedding_dim=self.d_model,
        #     padding_idx=None,
        #     max_norm=None, # changing max_norm to None -> will let model figure unless the training is unstable and we see exploding gradients.
        #     norm_type=2
        # )
        self.position_embedding = SinusoidalPositionalEmbeddings(d_model=self.d_model, max_seq_len=self.context_length)
        self.transformer_layers = torch.nn.Sequential()
        self.final_layer_norm = TorchLayerNorm(normalized_shape=self.d_model)
        # add sequential layers
        for layer in range(self.n_layers):
            self.transformer_layers.append(
                TransformerBlock(
                    d_model=self.d_model, 
                    n_heads=self.n_heads, 
                    scaling_factor=1/np.sqrt(2*self.n_layers), 
                    context_length=self.context_length,
                    attention_dropout=0.1,
                    logger=self.logger
                )
            )
        # add final layer normalization
        self.transformer_layers.append(self.final_layer_norm)
        # predict token with softmax
        # self.transformer_layers.append(torch.nn.Linear(in_features=self.d_model, out_features=self.vocab_size))
        self.dropout = torch.nn.Dropout(p=0.1)
        self.initialize_model_parameters(scaling_factor=1/np.sqrt(2*self.n_layers))
        self.print_param_stats()

    def print_param_stats(self, include_bias=True, filter_fn=None):
        """
        Print standard deviation of model parameters.

        Args:
            include_bias (bool): Whether to include bias terms.
            filter_fn (callable): Optional function (name, param) -> bool
                                to filter parameters.
        """
        self.logger.info(f"{'Layer Name':<60} | {'Std Dev':<10}")
        self.logger.info("-" * 75)

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Default filtering logic
            if not include_bias and 'bias' in name:
                continue

            # Custom filter override
            if filter_fn is not None and not filter_fn(name, param):
                continue

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
            
            # MISTAKE - weights were getting overwritten to 0.02 with linear layers as there was no dedicated loop for transformer blocks.
            for module in self.modules():        
                if isinstance(module, TransformerBlock):
                    # scale only residual connection weights correctly
                    torch.nn.init.normal_(module.MHA.in_proj_weight, mean=0.0, std=0.02)
                    torch.nn.init.normal_(module.MHA.out_proj.weight, mean=0.0, std=0.02)
                    self.logger.info("Initializing TransformerBlock MHA in_proj_weight with normal distribution")
                    self.logger.info(f"module.MHA.out_proj.weight.std() before scaling = {module.MHA.out_proj.weight.std().item()}")
                    self.logger.info(f"module.FFN.linear_projection.weight.std() before scaling = {module.FFN.linear_projection.weight.std().item()}")
                    
                    module.MHA.out_proj.weight.mul_(scaling_factor)
                    module.FFN.linear_projection.weight.mul_(scaling_factor)
                    self.logger.info(f"module.MHA.out_proj.weight.std() after scaling = {module.MHA.out_proj.weight.std().item()}")
                    self.logger.info(f"module.FFN.linear_projection.weight.std() after scaling = {module.FFN.linear_projection.weight.std().item()}")
                    self.logger.info(f"Scaling TransformerBlock MHA out_proj.weight and FFN linear_projection.weight by scaling_factor = {scaling_factor}")


    def forward(self, x, return_proba = False):
        # self.logger.debug(f"x.shape = {x.shape} | x.device = {x.device} | x.dtype = {x.dtype}")
        batch_size, seq_len = x.shape
        assert seq_len <= self.context_length, "Sequence length exceeds model context_length"
        x_learnt_embeddings = self.embedding(x)
        # self.logger.debug(f"x_learnt_embeddings.shape = {x_learnt_embeddings.shape} | x_learnt_embeddings.device = {x_learnt_embeddings.device} | x_learnt_embeddings.dtype = {x_learnt_embeddings.dtype}")
        x_pos_embeddings = self.position_embedding(x)
        # x_pos_embeddings = self.learnt_position_embedding(torch.arange(start=0, end=seq_len, device=x.device)).unsqueeze(0)
        # self.logger.debug(f"x_pos_embeddings.shape = {x_pos_embeddings.shape} | x_pos_embeddings.device = {x_pos_embeddings.device} | x_pos_embeddings.dtype = {x_pos_embeddings.dtype}")
        
        x_embeddings = x_learnt_embeddings + x_pos_embeddings
        # x_embeddings = torch.nn.functional.dropout(input=x_embeddings, p=0.1) # MISTAKE - I had initially used functional dropout here w/o train v/s inference mode check. Moving it to Dropout module which internally manages train v/s inference mode and also makes code cleaner.
        x_embeddings = self.dropout(x_embeddings)
        x_logits = self.transformer_layers(x_embeddings)
        # self.logger.debug(f"z.shape = {x_logits.shape} | z.device = {x_logits.device} | z.dtype = {x_logits.dtype}")
        x_logits = x_logits@self.embedding.weight.T
        # self.logger.debug(f"x_logits.shape = {x_logits.shape} | x_logits.device = {x_logits.device} | x_logits.dtype = {x_logits.dtype}")
        if return_proba:
            return torch.nn.functional.softmax(input=x_logits, dim=-1)
        else:
            return x_logits