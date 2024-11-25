# MDL: Variational MDL probing

## Overview
Probing implementation via MDL approach, using variational dropout instead of last linear layer. 
That is theoretically equivalent to learning parameters of the model, minimizing the total description lengths of the data encoded with that model and model parameters.

---

## 📖 Theoretical Background

See 2.2.1 Variatonal Code in original [paper](https://arxiv.org/pdf/2003.12298).

<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.14.305/pdf.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.14.305/pdf.worker.min.js"></script>

<div id="pdf-container" style="height: 600px; overflow: auto;"></div>
<script>
  const url = 'https://arxiv.org/pdf/2003.12298';

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
Detailed documentation of the MLP (Multi-Layer Perceptron) model used for variational modeling.

::: problib.mdl.variational_probing
    handler: python

---

### 🎯 Training and Probing
Comprehensive guide on the trainer used to probe and train the variational model.

::: problib.mdl.Trainer
    handler: python

::: problib.mdl.BayesianLayers
    handler: python

---

### 🛠️ Utils

::: problib.mdl.utils
    handler: python

---

## 🚀 Usage Guide

Below is a step-by-step usage example of the Variational MDL probing for POS-tagging task

```python
from problib.mdl.variational_probing import VariationalProbingModel
from problib.mdl.Trainer import Trainer
from torch.utils.data import DataLoader
from utils import *
from transformers import AutoTokenizer

# Define the model configuration with essential parameters
train_config = {
    "variational": True, 
    "eval_metrics": ["description_length"],
    "lr": 1e-3,
    "optimizer": "Adam",
    "n_epochs": 100,
    "loss_function": "crossentropy"
}

# Initialize the probing model with the defined configuration
model = VariationalProbingModel(
    pretrained_path="pretrained/model/path",
)
tokenizer = AutoTokenizer.from_pretrained("pretrained/model/path")

# Set up the trainer for model training
trainer = Trainer(
    model=model,
    train_config=train_config
)

# get the datasets
data_path_train = ...
data_path_val = ...
data_path_test = ...

train_dataset = MDLDataset_POSTagging(data_path_train)
val_dataset = MDLDataset_POSTagging(data_path_val)
test_dataset = MDLDataset_POSTagging(data_path_test)

# get the loaders
collator = Collator(
    tokenizer=tokenizer,
    max_length=1024,
    padding=True,
    truncation=True,
    add_special_tokens=True
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collator)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collator)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collator)

# Train the model
model = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    evaluate_every=10
)

# Get the evaluation results
val_metrics = trainer._metrics

# Get testing results
test_metrics = trainer.evaluate(test_loader)

```
