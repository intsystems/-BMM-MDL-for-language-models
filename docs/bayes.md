# MDL: Bayesian Probing

## Overview

Bayesian approach similar to other approaches applies information-theoretic concepts to streamline data compression and emphasize essential features. It utilises Bayesian mutual information from the perspective of Bayesian agents. For instance, under Bayesian MI various operations with data affect its representations. Similar to MDL, it can be applied to the problem of probing.

---

## 📖 Theoretical Background

You can dive into theoretical underpinnings in our [blogpost](https://www.overleaf.com/project/6728a4c896d75ac1f40faf4d), or dive into original [paper](https://aclanthology.org/2021.emnlp-main.229).


---

## 📚 Documentation

Explore the components of the model used in this approach, organized into sections for clarity.

### 🧠 Model Architecture
The Bayesian Probing Model integrates an MLP classifier with a sampling mechanism.

::: problib.bayesian.probing
    handler: python

::: problib.bayesian.bayesian_model
    handler: python

---

## 🚀 Usage Guide

Below is a step-by-step guide to implementing Bayesian Agents belief approach using an MLP configuration.

In the example, we will solve a task of POS-tagging, however it can be applied to classification task as well.

```python
from problib.base import BaseModel
from problib.bayesian.belief_agents import MLP
from problib.bayesian.probing import BayesianProbingModel

import MLPConfig, ProbingModel, MLPTrainer

# Define the model configuration with essential parameters
train_config = MLPConfig(
    K=5,  # Number of top features to sample
    input_dim=768,  # Input dimensionality
    hidden_dim=256,  # Size of hidden layers
    output_dim=10,  # Output dimensionality (e.g., number of classes)
    num_layers=2,  # Number of layers in the MLP
    sampler_type="poisson"  # Type of sampler, e.g., "poisson" or "conditional_poisson"
)

# Initialize the probing model with the defined configuration
model = BayesianProbingModel(belief_model,
                             train_config)

# Set up the trainer for model training
trainer = MLPTrainer(
    model=model,
    train_dataset=train_dataset,  # Training dataset
    mc_samples=5,  # Monte Carlo samples for likelihood estimation
    entropy_scale=1e-3,  # Regularization scale for entropy
    l1_weight=1e-5,  # Weight for L1 regularization
    l2_weight=1e-5  # Weight for L2 regularization
)

# Train the model
trainer.train()

# Get results of probing 
probing_results = model.probe.get_probe_results()
print("Probing results:", probing_results)
```
