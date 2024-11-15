# MDL: Variational MDL probing

## Overview
Probing implementation via MDL approach, using variational dropout instead of last linear layer. 
That is theoretically equivalent to learning parameters of the model, minimizing the total description lengths of the data encoded with that model and model parameters.

---

## 📖 Theoretical Background

See 2.2.1 Variatonal Code in original [paper](https://arxiv.org/pdf/2003.12298).

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
