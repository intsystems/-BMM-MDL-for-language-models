import torch
from torch import nn
from transformers import PretrainedConfig
from .samplers import get_sampler
from .base import BaseModel

class MLPConfig(PretrainedConfig):
    model_type = "mlp_classifier"
    
    def __init__(
        self, 
        K: int,  
        input_dim: int = 768, 
        hidden_dim: int = 256, 
        output_dim: int = 10, 
        num_layers: int = 2, 
        sampler_type: str = "poisson", 
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.sampler_type = sampler_type
        self.D = input_dim
        self.K = K


class MLPClassifier(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        layers = []
        input_dim = config.input_dim
        for _ in range(config.num_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            input_dim = config.hidden_dim
        layers.append(nn.Linear(config.hidden_dim, config.output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class ProbingModel(BaseModel):
    name = 'probing_model'

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.sampler = get_sampler(config.sampler_type, config.D, config.K)
        self.classifier = MLPClassifier(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.sampler.sample(x.size(0)) 
        masked_x = x * mask 
        logits = self.classifier(masked_x)
        return logits
