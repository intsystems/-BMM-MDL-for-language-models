from BayesianLayers import *
from torch import nn
from transformers import AutoModel, AutoTokenizer


class VariationalProbingModel(nn.Module):
    def __init__(self, pretrained_path="D:/models/roberta-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_path)
        self.model.pooler.dense = LinearGroupNJ(
            in_features=self.model.pooler.dense.in_features,
            out_features=self.model.pooler.dense.out_features,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def eval(self):
        raise NotImplementedError
