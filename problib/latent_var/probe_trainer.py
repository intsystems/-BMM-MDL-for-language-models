from transformers import Trainer
from torch import nn

class MLPTrainer(Trainer):
    def __init__(self, model, sampler, *args, mc_samples=5, entropy_scale=1e-3, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.sampler = sampler
        self.mc_samples = mc_samples
        self.entropy_scale = entropy_scale
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        batch_size = inputs["input_ids"].shape[0]
        
        total_loss = 0
        for _ in range(self.mc_samples):
            mask = self.sampler.sample() 
            masked_inputs = inputs["input_ids"] * mask.to(inputs["input_ids"].device)
            
            outputs = model(masked_inputs)
            logits = outputs.logits
            
            loss_fn = nn.CrossEntropyLoss()
            loss_mc = loss_fn(logits, labels)
            total_loss += loss_mc
        
        total_loss /= self.mc_samples
        
        loss_entropy = -self.sampler.weights.softmax(dim=0).log().mean() 
        total_loss += self.entropy_scale * loss_entropy
        
        return (total_loss, outputs) if return_outputs else total_loss