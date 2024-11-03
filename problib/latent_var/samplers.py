import torch
from torch import nn

class BaseSampler(nn.Module):
    def __init__(self, D, K):
        super().__init__()
        self.D = D
        self.K = K
    
    def sample(self):
        raise NotImplementedError("Each sampler must implement the sample method.")

class ConditionalPoissonSampler(BaseSampler):
    def __init__(self, D, K):
        super().__init__(D, K)
        self.weights = nn.Parameter(torch.rand(D))

    def sample(self):
        mask = torch.zeros(self.D, dtype=torch.bool)
        return mask

class PoissonSampler(BaseSampler):
    def __init__(self, D, K):
        super().__init__(D, K)
        self.weights = nn.Parameter(torch.rand(D))

    def sample(self):
        mask = torch.bernoulli(self.weights).bool()
        return mask

def get_sampler(sampler_type, D, K):
    if sampler_type == "conditional_poisson":
        return ConditionalPoissonSampler(D, K)
    elif sampler_type == "poisson":
        return PoissonSampler(D, K)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")