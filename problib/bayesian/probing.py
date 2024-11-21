from ..base import BaseModel
from bayesian_model import MLP
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import os
import sys
import argparse
from tqdm import tqdm
import torch.optim as optim

sys.path.insert(1, os.path.join(sys.path[0], '..'))

class BayesianProbingModel(BaseModel):
    """
    Bayesian Probing Model.
    """
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
        """
        Initialize the model.
        Parameters:
            embedding_size (int): The size of the embedding.
            n_classes (int): The number of classes.
            hidden_size (int): The size of the hidden layer.
            n_layers (int): The number of layers.
            dropout (float): The dropout rate.
        """ 
        super().__init__()
        self.model = MLP(embedding_size=embedding_size,
                                    n_classes=n_classes,
                                    hidden_size=hidden_size,
                                    nlayers=n_layers,
                                    dropout=dropout,
                                    representation=representation,
                                    n_words=n_words)
        self.model.to(device)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.
        Parameters:
            input_ids (torch.Tensor): The input ids of the tokens.
            attention_mask (torch.Tensor): The attention mask of the tokens.
        Returns:
            torch.Tensor: The output of the model.
        """ 
        if attention_mask:
            return self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.model(input_ids=input_ids)

    def _evaluate(self, evalloader, model):
        """
        Evaluate the model on the evaluation dataset.
        Parameters:
            evalloader (DataLoader): The evaluation dataset.
            model (BayesianProbingModel): The model to evaluate.
        Returns:
            dict: A dictionary containing the evaluation loss and accuracy.
        """
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


    def evaluate(self, evalloader, model):
        """
        Evaluate the model on the evaluation dataset.
        Parameters:
            evalloader (DataLoader): The evaluation dataset.
            model (BayesianProbingModel): The model to evaluate.
        Returns:
            dict: A dictionary containing the evaluation loss and accuracy.
        """
        model.eval()
        with torch.no_grad():
            result = self._evaluate(evalloader, model)
        model.train()
        return result

    def train_epoch(self, trainloader, devloader, model, optimizer, train_info):
        """
        Train the model for one epoch.
        Parameters:
            trainloader (DataLoader): The training dataset.
            devloader (DataLoader): The development dataset.
            model (BayesianProbingModel): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            train_info (TrainInfo): The training information.
        """
        for x, y in trainloader:
            loss = model.train_batch(x, y, optimizer)
            train_info.new_batch(loss)
    
            if train_info.eval:
                dev_results = self.evaluate(devloader, model)
    
                if train_info.is_best(dev_results):
                    model.set_best()
                elif train_info.finish:
                    train_info.print_progress(dev_results)
                    return
    
                train_info.print_progress(dev_results)

    def train(self, trainloader, devloader, model, eval_batches, wait_iterations):
        optimizer = optim.AdamW(model.parameters())
    
        with tqdm(total=wait_iterations) as pbar:
            train_info = TrainInfo(pbar, wait_iterations, eval_batches)
            while not train_info.finish:
                self.train_epoch(trainloader, devloader, model,
                            optimizer, train_info)
    
        model.recover_best()