# MDL: Variational MDL probing

## Overview
Probing implementation via MDL approach, using variational dropout instead of last linear layer. 
That is theoretically equivalent to learning parameters of the model, minimizing the total description lengths of the data encoded with that model and model parameters.

---

## 📖 Theoretical Background

See 2.2.1 Variatonal Code in original [paper](https://arxiv.org/pdf/2003.12298).

---

## 🚀 Usage Guide

Below is a step-by-step guide to implementing Bayesian Agents belief approach using an MLP configuration.

In the example, we will solve a task of POS-tagging, however it can be applied to classification task as well.

```python
from problib.mdl.variational_probing import VariationalProbingModel
from problib.mdl.Trainer import Trainer

# Define the model configuration with essential parameters
train_config = #TODO

# Initialize the probing model with the defined configuration
model = VariationalProbingModel(
  pretrained_path="pretrained/model/path",
)

# Set up the trainer for model training
trainer = Trainer(
    model=model,
    train_config=train_config
)

# Train the model
trainer.train(train_loader, val_loader)

# Get results of probing 
probing_results = trainer.evaluate(test_loader)
print("Probing results:", probing_results)
```
