from transformers import Trainer
from torch import nn
from .modeling_mlp import ProbeModel  
import torch

class MLPTrainer(Trainer):
    def __init__(self, model: ProbeModel, *args, mc_samples: int = 5, entropy_scale: float = 1e-3, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.mc_samples = mc_samples
        self.entropy_scale = entropy_scale

    def compute_loss(self, model: ProbeModel, inputs: dict, return_outputs: bool = False) -> torch.Tensor:
        labels = inputs.get("labels")
        input_ids = inputs["input_ids"]
        
        total_loss = 0
        for _ in range(self.mc_samples):
            outputs, mask = model(input_ids)

            loss_fn = nn.CrossEntropyLoss()
            loss_mc = loss_fn(outputs, labels)
            total_loss += loss_mc

        total_loss /= self.mc_samples

        loss_entropy = -model.sampler.weights.softmax(dim=0).log().mean()
        total_loss += self.entropy_scale * loss_entropy
        
        return (total_loss, outputs) if return_outputs else total_loss
