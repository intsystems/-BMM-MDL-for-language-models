# MDL: Latent Variable Probing

## Overview

Approach 1 applies information-theoretic concepts to streamline data compression and emphasize essential features. Rooted in MDL (Minimum Description Length) theory, this approach is highly effective for managing large datasets and performing feature reduction. By concentrating on the most impactful features, it enhances model efficiency, making it particularly valuable for complex, high-dimensional data.

---

## 📖 Theoretical Background

Put description 

---

## 📄 Original Research


For an in-depth exploration of the theoretical underpinnings, you can view the paper directly below:

<iframe src="https://arxiv.org/pdf/2201.08214" width="100%" height="600px">
    This browser does not support PDFs. Please download the PDF to view it: 
    <a href="https://arxiv.org/pdf/2201.08214">Download PDF</a>.
</iframe>

---

## 🚀 Usage Guide

Below is a step-by-step guide to implementing Latent Variable approach using an MLP configuration.

```python
from mymodule import MLPConfig, ProbingModel, MLPTrainer

# Define the model configuration with essential parameters
config = MLPConfig(
    K=5,  # Number of top features to sample
    input_dim=768,  # Input dimensionality
    hidden_dim=256,  # Size of hidden layers
    output_dim=10,  # Output dimensionality (e.g., number of classes)
    num_layers=2,  # Number of layers in the MLP
    sampler_type="poisson"  # Type of sampler, e.g., "poisson" or "conditional_poisson"
)

# Initialize the probing model with the defined configuration
model = ProbingModel(config)

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

# Retrieve the top K features based on learned sampler weights
top_k_features = model.sampler.get_top_k_features()
print("Top K features:", top_k_features)
```
