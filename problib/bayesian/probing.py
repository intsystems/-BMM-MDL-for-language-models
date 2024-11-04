from ..base import BaseModel
from bayesian_model import MLP
from torch import nn
from transformers import AutoModel, AutoTokenizer
import os
import sys
import argparse
from tqdm import tqdm
import torch.optim as optim

sys.path.insert(1, os.path.join(sys.path[0], '..'))

class VariationalProbingModel(BaseModel):
    def __init__(
        self,
        embedding_size = 512,
        n_classes = 2,
        hidden_size = 512,
        n_layers = 10,
        dropout = 0.1,
        representation = None,
        n_words=10,
        device='cuda'
        
    ):  
        super().__init__()
        self.model = MLP(embedding_size=embedding_size,
                                    n_classes=n_classes,
                                    hidden_size=hidden_size,
                                    nlayers=nlayers,
                                    dropout=dropout,
                                    representation=representation,
                                    n_words=n_words)
        self.model.to(device)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids)

    def _evaluate(evalloader, model):
        dev_loss, dev_acc = 0, 0
        for x, y in evalloader:
            loss, acc = model.eval_batch(x, y)
            dev_loss += loss
            dev_acc += acc
    
        n_instances = len(evalloader.dataset)
        return {
            'loss': dev_loss / n_instances,
            'acc': dev_acc / n_instances,
        }


    def evaluate(evalloader, model):
        model.eval()
        with torch.no_grad():
            result = _evaluate(evalloader, model)
        model.train()
        return result

    def train_epoch(trainloader, devloader, model, optimizer, train_info):
        for x, y in trainloader:
            loss = model.train_batch(x, y, optimizer)
            train_info.new_batch(loss)
    
            if train_info.eval:
                dev_results = evaluate(devloader, model)
    
                if train_info.is_best(dev_results):
                    model.set_best()
                elif train_info.finish:
                    train_info.print_progress(dev_results)
                    return
    
                train_info.print_progress(dev_results)

    def train(trainloader, devloader, model, eval_batches, wait_iterations):
        optimizer = optim.AdamW(model.parameters())
    
        with tqdm(total=wait_iterations) as pbar:
            train_info = TrainInfo(pbar, wait_iterations, eval_batches)
            while not train_info.finish:
                train_epoch(trainloader, devloader, model,
                            optimizer, train_info)
    
        model.recover_best()



