from typing import Optional, Tuple, Dict, Any, List, Union
from transformers import Trainer
from torch import nn
import torch

class MLPTrainer(Trainer):
    def __init__(
        self, 
        model: nn.Module, 
        mc_samples: int = 5, 
        entropy_scale: float = 1e-3, 
        l1_weight: float = 0.0, 
        l2_weight: float = 0.0, 
        *args, 
        **kwargs
    ):
        super().__init__(model=model, *args, **kwargs)
        self.mc_samples = mc_samples
        self.entropy_scale = entropy_scale
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sampler = model.sampler  # Use sampler from the model

    def compute_loss(
        self, 
        model: nn.Module, 
        inputs: Dict[str, torch.Tensor], 
        return_outputs: Optional[bool] = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        labels = inputs.get("labels")
        input_ids = inputs["input_ids"]
        batch_size = input_ids.shape[0]

        loss_fn = nn.CrossEntropyLoss(reduction="none")
        
        total_loss_mc = torch.zeros(batch_size, device=input_ids.device)
        
        log_cache = [] if hasattr(self.sampler, 'logZ') and 'log_cache' in self.sampler.logZ.__code__.co_varnames else None
        logZ = self.sampler.logZ(cache=log_cache) if log_cache is not None else self.sampler.logZ()

        for _ in range(self.mc_samples):
            if log_cache is not None:
                mask = self.sampler.sample(batch_size, log_cache=log_cache)
            else:
                mask = self.sampler.sample(batch_size)
            
            logits = model(input_ids, mask)
            loss_mc = loss_fn(logits, labels)
            total_loss_mc += loss_mc
        
        loss_mc = total_loss_mc / self.mc_samples
        loss_mc = loss_mc.sum() / batch_size

        total_loss_rf = torch.zeros(batch_size, device=input_ids.device)
        
        for _ in range(self.mc_samples):
            if log_cache is not None:
                mask = self.sampler.sample(batch_size, log_cache=log_cache).detach()
            else:
                mask = self.sampler.sample(batch_size).detach()

            logits = model(input_ids, mask)
            reward = loss_fn(logits, labels).detach()
            logprob = self.sampler.logprob(mask, logZ=logZ) if log_cache is not None else self.sampler.logprob(mask)
            total_loss_rf += reward * logprob

        loss_rf = total_loss_rf / self.mc_samples
        loss_rf = loss_rf.sum() / batch_size

        loss_entropy = -self.sampler.entropy(cache=log_cache) if log_cache is not None else -self.sampler.entropy()

        weights_regularization = 0.0
        for p in model.parameters():
            weights_regularization += self.l1_weight * p.abs().sum() + self.l2_weight * (p ** 2).sum()

        loss = loss_mc + self.entropy_scale * loss_entropy + loss_rf + weights_regularization

        return (loss, logits) if return_outputs else loss
