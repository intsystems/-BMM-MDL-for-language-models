import math
import torch
import torch.nn as nn

from ..base import BaseModel


class MLP(BaseModel):
    name = 'mlp'

    def __init__(self, task, embedding_size=768, n_classes=3, hidden_size=5,
                 nlayers=1, dropout=0.1, representation=None, n_words=None):
        # pylint: disable=too-many-arguments
        super().__init__()

        # Save things to the model here
        self.dropout_p = dropout
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.n_classes = n_classes
        self.representation = representation
        self.n_words = n_words
        self.task = task

        if self.representation in ['onehot', 'random']:
            self.build_embeddings(n_words, embedding_size)

        self.mlp = self.build_mlp()
        self.out = nn.Linear(self.final_hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)

        self.criterion = nn.CrossEntropyLoss()

    def build_embeddings(self, n_words, embedding_size):
        if self.task == 'dep_label':
            self.embedding_size = int(embedding_size / 2) * 2
            self.embedding = nn.Embedding(n_words, int(embedding_size / 2))
        else:
            self.embedding = nn.Embedding(n_words, embedding_size)

        if self.representation == 'random':
            self.embedding.weight.requires_grad = False

    def build_mlp(self):
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
        if self.representation in ['onehot', 'random']:
            x = self.get_embeddings(x)

        x_emb = self.dropout(x)
        x = self.mlp(x_emb)
        logits = self.out(x)
        return logits

    def get_embeddings(self, x):
        x_emb = self.embedding(x)
        if len(x.shape) > 1:
            x_emb = x_emb.reshape(x.shape[0], -1)

        return x_emb

    def train_batch(self, data, target, optimizer):
        optimizer.zero_grad()
        mlp_out = self(data)
        loss = self.criterion(mlp_out, target)
        loss.backward()
        optimizer.step()

        return loss.item() / math.log(2)

    def eval_batch(self, data, target):
        mlp_out = self(data)
        loss = self.criterion(mlp_out, target) / math.log(2)
        accuracy = (mlp_out.argmax(dim=-1) == target).float().detach().sum()
        loss = loss.item() * data.shape[0]

        return loss, accuracy

    @staticmethod
    def get_norm():
        return torch.Tensor([0])

    def get_args(self):
        return {
            'nlayers': self.nlayers,
            'hidden_size': self.hidden_size,
            'embedding_size': self.embedding_size,
            'dropout': self.dropout_p,
            'n_classes': self.n_classes,
            'representation': self.representation,
            'n_words': self.n_words,
            'task': self.task,
        }

    @staticmethod
    def print_param_names():
        return [
            'n_layers', 'hidden_size', 'embedding_size', 'dropout',
            'n_classes', 'representation', 'n_words',
        ]

    def print_params(self):
        return [
            self.nlayers, self.hidden_size, self.embedding_size, self.dropout_p,
            self.n_classes, self.representation, self.n_words
        ]


class Categoric(BaseModel):
    name = 'categoric'

    def __init__(self, task, n_classes=100):
        super().__init__()

        self.task = task
        self.n_classes = n_classes
        self.alpha = 2

        self.probs = nn.Parameter(torch.Tensor(self.n_classes))
        self.log_probs = nn.Parameter(torch.Tensor(self.n_classes))
        self.count = nn.Parameter(
            torch.LongTensor(self.n_classes).zero_(),
            requires_grad=False)

        self.criterion = nn.NLLLoss(ignore_index=self.ignore_index)

    def fit(self, trainloader):
        with torch.no_grad():
            for _, y in trainloader:
                self.fit_batch(_, y)

    def fit_batch(self, _, y):
        for char in y.unique():
            if char == self.ignore_index:
                continue
            self.count[char] += (y == char).sum()

        self.probs[:] = \
            (self.count.float() + self.alpha) / (self.count.sum() + self.alpha * self.n_classes)
        self.log_probs[:] = torch.log(self.probs)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.task == 'parse':
            max_len = x.shape[1]
            y_hat = self.log_probs[:max_len] \
                .reshape(1, 1, -1) \
                .repeat(batch_size, max_len, 1)
        else:
            y_hat = self.log_probs \
                .reshape(1, -1) \
                .repeat(batch_size, 1)

        return y_hat

    def eval_batch(self, data, target):
        mlp_out = self(data)
        loss = self.criterion(mlp_out, target) / math.log(2)
        accuracy = (mlp_out.argmax(dim=-1) == target).float().detach().sum()
        loss = loss.item() * data.shape[0]

        return loss, accuracy

    def get_args(self):
        return {
            'n_classes': self.n_classes,
            'task': self.task,
        }

    @staticmethod
    def print_param_names():
        return [
            'n_classes', 'task'
        ]

    def print_params(self):
        return [
            self.n_classes, self.task
        ]
