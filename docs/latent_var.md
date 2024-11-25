# MDL: Latent Variable Probing

## Overview

Approach 1 applies information-theoretic concepts to streamline data compression and emphasize essential features. Rooted in MDL (Minimum Description Length) theory, this approach is highly effective for managing large datasets and performing feature reduction. By concentrating on the most impactful features, it enhances model efficiency, making it particularly valuable for complex, high-dimensional data.

---

## 📖 Theoretical Background

For an in-depth exploration of the theoretical underpinnings, you can view the [paper](https://arxiv.org/pdf/2201.08214):

<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.14.305/pdf.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.14.305/pdf.worker.min.js"></script>

<div id="pdf-container" style="height: 600px; overflow: auto;"></div>
<script>
  const url = 'https://arxiv.org/pdf/2201.08214';

  const pdfjsLib = window['pdfjs-dist/build/pdf'];
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.14.305/pdf.worker.min.js';

  const container = document.getElementById('pdf-container');

  pdfjsLib.getDocument(url).promise.then((pdf) => {
    console.log(`Total pages: ${pdf.numPages}`);
    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
      pdf.getPage(pageNum).then((page) => {
        const viewport = page.getViewport({ scale: 1.5 }); // Adjust scale for better quality
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.height = viewport.height;
        canvas.width = viewport.width;

        container.appendChild(canvas);

        const renderContext = { canvasContext: context, viewport: viewport };
        page.render(renderContext);
      });
    }
  });
</script>



---

## 📚 Documentation

Explore the components of the model used in this approach, organized into sections for clarity.

### 🧠 Model Architecture
Detailed documentation of the MLP (Multi-Layer Perceptron) model used for latent variable modeling.

::: problib.latent_var.modeling_mlp
    handler: python

---

### 🎯 Training and Probing
Comprehensive guide on the trainer used to probe and train the latent variable model.

::: problib.latent_var.probe_trainer
    handler: python

---

### 🎲 Samplers
Overview of the sampling methods integrated into this approach.

::: problib.latent_var.samplers
    handler: python

---

## 🚀 Usage Guide

Below is a step-by-step guide to implementing Latent Variable approach using an MLP configuration.

```python
from problib.latent_var import MLPConfig, ProbingModel, MLPTrainer

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
