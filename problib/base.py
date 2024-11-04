from abc import ABC, abstractmethod
import copy
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    # pylint: disable=abstract-method
    name = 'base'
    ignore_index = -1

    def __init__(self):
        super().__init__()

        self.best_state_dict = None
	self.model = None)

    def set_best(self):
        self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        self.load_state_dict(self.best_state_dict)

    def save(self, path):
        fname = self.get_name(path)
        torch.save({
            'kwargs': self.get_args(),
            'model_state_dict': self.state_dict(),
        }, fname)

    @abstractmethod
    def get_args(self):
        pass

    @classmethod
    def load(cls, path):
        checkpoints = cls.load_checkpoint(path)
        self.model = cls(**checkpoints['kwargs'])
        self.model.load_state_dict(checkpoints['model_state_dict'])
        return self.model

    @classmethod
    def load_checkpoint(cls, path):
        fname = cls.get_name(path)
        return torch.load(fname, map_location=constants.device)

    @classmethod
    def get_name(cls, path):
        return '%s/model.tch' % (path)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def eval(self):
	raise NotImplementedError
