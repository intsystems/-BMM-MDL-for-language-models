from BayesianLayers import *
from torch import nn
from transformers import AutoModel, AutoTokenizer


class RegularProbingModel(nn.Module):
    def __init__(self, pretrained_path="D:/models/roberta-base", out_features=None):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_path)
        if out_features is None:
            out_features = self.model.pooler.dense.out_features
        self.probing_layer = nn.Linear(self.model.pooler.dense.in_features, out_features)

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            backbone_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.probing_layer(backbone_outputs.last_hidden_state)


class VariationalProbingModel(RegularProbingModel):
    def __init__(self, pretrained_path="D:/models/roberta-base", out_features=None):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_path)
        if out_features is None:
            out_features = self.model.pooler.dense.out_features
        self.probing_layer = LinearGroupNJ(self.model.pooler.dense.in_features, out_features)
    
    def kl_divergence(self):
        return self.probing_layer.kl_divergence()

    def train(self, *args, **kwargs):
        self.probing_layer.deterministic = False
        return super().train(*args, **kwargs)
    
    def eval(self, *args, **kwargs):
        self.probing_layer.deterministic = True
        return super().eval(*args, **kwargs)
    
