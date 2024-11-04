from .. import BaseMoDel
from torch import nn
from transformers import AutoModel, AutoTokenizer


class VariationalProbingModel(BaseModel):
    def __init__(
        self
    ):  
        super().__init__()
        self.model = model = MLP(
            args.task, embedding_size=args.embedding_size,
            n_classes=n_classes, hidden_size=args.hidden_size,
            nlayers=args.nlayers, dropout=args.dropout,
            representation=representation, n_words=n_words)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def eval(self):
        raise NotImplementedError


