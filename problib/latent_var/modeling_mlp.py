import torch
from torch import nn
from transformers import PretrainedConfig

class MLPConfig(PretrainedConfig):
    model_type = "mlp_classifier"
    
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=10, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

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
    

class ProbeModel(nn.Module):
    def __init__(self, classifier: nn.Module, sampler: nn.Module):
        super().__init__()
        self.classifier = classifier
        self.sampler = sampler
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.sampler.sample().to(x.device)
        masked_x = x * mask  
        output = self.classifier(masked_x)
        return output, mask
