import math
import torch
import torch.nn as nn

from .base import BaseModel


class MLP(BaseModel):
    """
    Multi-layer Perceptron model.
    """

    name = "mlp"

    def __init__(
        self,
        task,
        embedding_size=768,
        n_classes=3,
        hidden_size=5,
        nlayers=1,
        dropout=0.1,
        representation=None,
        n_words=None,
    ):
        """
        Initialize the model.
        Parameters:
            task (str): The task to perform.
            embedding_size (int): The size of the embedding.
            n_classes (int): The number of classes.
            hidden_size (int): The size of the hidden layer.
            nlayers (int): The number of layers.
            dropout (float): The dropout rate.
            representation (str): The representation to use.
            n_words (int): The number of words.
        """
        super().__init__()

        self.dropout_p = dropout
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.n_classes = n_classes
        self.representation = representation
        self.n_words = n_words
        self.task = task

        if self.representation in ["onehot", "random"]:
            self.build_embeddings(n_words, embedding_size)

        self.mlp = self.build_mlp()
        self.out = nn.Linear(self.final_hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)

        self.criterion = nn.CrossEntropyLoss()

    def build_embeddings(self, n_words, embedding_size):
        """
        Build the embeddings for the model.
        Parameters:
            n_words (int): The number of words.
            embedding_size (int): The size of the embedding.
        """
        if self.task == "dep_label":
            self.embedding_size = int(embedding_size / 2) * 2
            self.embedding = nn.Embedding(n_words, int(embedding_size / 2))
        else:
            self.embedding = nn.Embedding(n_words, embedding_size)

        if self.representation == "random":
            self.embedding.weight.requires_grad = False

    def build_mlp(self):
        """
        Build the MLP for the model.
        """
        if self.nlayers == 0:
            self.final_hidden_size = self.embedding_size
            return nn.Identity()

        src_size = self.embedding_size
        tgt_size = self.hidden_size
        mlp = []
        for _ in range(self.nlayers):
            mlp += [nn.Linear(src_size, tgt_size)]
            mlp += [nn.ReLU()]
            mlp += [nn.Dropout(self.dropout_p)]
            src_size, tgt_size = tgt_size, int(tgt_size / 2)
        self.final_hidden_size = src_size
        return nn.Sequential(*mlp)

    def forward(self, x):
        """
        Forward pass of the model.
        Parameters:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor.
        """
        if self.representation in ["onehot", "random"]:
            x = self.get_embeddings(x)

        x_emb = self.dropout(x)
        x = self.mlp(x_emb)
        logits = self.out(x)
        return logits

    def get_embeddings(self, x):
        """
        Get the embeddings for the model.
        Parameters:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor.
        """
        x_emb = self.embedding(x)
        if len(x.shape) > 1:
            x_emb = x_emb.reshape(x.shape[0], -1)

        return x_emb

    def train_batch(self, data, target, optimizer):
        """
        Train the model for one batch.
        Parameters:
            data (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.
            optimizer (torch.optim.Optimizer): The optimizer to use.
        Returns:
            float: The loss.
        """
        optimizer.zero_grad()
        mlp_out = self(data)
        loss = self.criterion(mlp_out, target)
        loss.backward()
        optimizer.step()

        return loss.item() / math.log(2)

    def eval_batch(self, data, target):
        """
        Evaluate the model for one batch.
        Parameters:
            data (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.
        Returns:
            tuple: A tuple containing the loss and accuracy.
        """
        mlp_out = self(data)
        loss = self.criterion(mlp_out, target) / math.log(2)
        accuracy = (mlp_out.argmax(dim=-1) == target).float().detach().sum()
        loss = loss.item() * data.shape[0]

        return loss, accuracy

    @staticmethod
    def get_norm():
        """
        Get the norm of the model.
        Returns:
            torch.Tensor: The norm of the model.
        """
        return torch.Tensor([0])

    def get_args(self):
        """
        Get the arguments for the model.
        Returns:
            dict: A dictionary containing the arguments.
        """
        return {
            "nlayers": self.nlayers,
            "hidden_size": self.hidden_size,
            "embedding_size": self.embedding_size,
            "dropout": self.dropout_p,
            "n_classes": self.n_classes,
            "representation": self.representation,
            "n_words": self.n_words,
            "task": self.task,
        }

    @staticmethod
    def print_param_names():
        """
        Print the parameter names.
        Returns:
            list: A list of parameter names.
        """
        return [
            "n_layers",
            "hidden_size",
            "embedding_size",
            "dropout",
            "n_classes",
            "representation",
            "n_words",
        ]

    def print_params(self):
        """
        Print the parameters.
        Returns:
            list: A list of parameters.
        """
        return [
            self.nlayers,
            self.hidden_size,
            self.embedding_size,
            self.dropout_p,
            self.n_classes,
            self.representation,
            self.n_words,
        ]
