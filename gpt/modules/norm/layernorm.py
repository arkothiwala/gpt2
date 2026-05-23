import torch
class CustomLayerNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.weight = torch.nn.Parameter(data=torch.ones(size=(self.d_model,)))
        self.bias = torch.nn.Parameter(data=torch.zeros(size=(self.d_model,)))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False because we want to divide by N instead of N-1, because we are calculating the variance for the entire population (the sequence) and not a sample of the population.
        # torch by default uses unbiased=True, which divides by N-1, so we need to set it to False to divide by N. Leading to incorrect results and mismatch with torch.nn.LayerNorm results.
        # MISTAKE - initially I had not set it to false there for results weren't matching with torch.nn.LayerNorm
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x-mean)/torch.sqrt(var + self.eps)
        x_normalized = (x_normalized*self.weight)+self.bias
        return x_normalized
