from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torch

# TODO: implement losses?
# TODO: implement datasets and dataloaders
# TODO: implement MDL calculation
# TODO: implement metrics


class MDLDataset_Base(Dataset):
    def __init__(
        self,
        data_path,
    ):
        super().__init__()
        self.data = self._read_data(data_path)

    def _read_data(self, data_path):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __get_item__(self, idx):
        raise NotImplementedError


class MDLDataset_POSTagging(MDLDataset_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __get_item__(self, idx):
        raise NotImplementedError


class Collator:
    def __init__(
        self,
        tokenizer,
        max_length=512,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        **tokenizer_kwargs
    ):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        if self.tokenizer_kwargs.get("max_length", None) is None:
            self.tokenizer_kwargs["max_length"] = max_length
        if self.tokenizer_kwargs.get("padding", None) is None:
            self.tokenizer_kwargs["padding"] = padding
        if self.tokenizer_kwargs.get("truncation", None) is None:
            self.tokenizer_kwargs["truncation"] = truncation
        if self.tokenizer_kwargs.get("add_special_tokens", None) is None:
            self.tokenizer_kwargs["add_special_tokens"] = add_special_tokens

    def __call__(self, batch):
        texts = [elem["text"] for elem in batch]
        labels = [elem["label"] for elem in batch]
        tokenized = self.tokenizer(texts, return_tensors="pt", **self.tokenizer_kwargs)
        labels = torch.tensor(labels, dtype=torch.long)

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        return input_ids, attention_mask, labels
