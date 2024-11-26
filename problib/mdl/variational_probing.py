from BayesianLayers import *
from torch import nn
from transformers import AutoModel, AutoTokenizer


class VariationalProbingModel(nn.Module):
    """
    A variational probing model built on a pre-trained transformer backbone with a Bayesian probing layer.

    This model integrates a `LinearGroupNJ` layer as the probing layer to add a Bayesian component to the architecture.

    Args:
        pretrained_path (str, optional): Path to the pre-trained transformer model. Defaults to "D:/models/roberta-base".
        out_features (int, optional): Number of output features for the probing layer. Defaults to the output dimension
            of the pooler dense layer in the backbone model.

    Attributes:
        model (transformers.AutoModel): The pre-trained transformer model as the backbone.
        probing_layer (LinearGroupNJ): The Bayesian probing layer.
    """

    def __init__(self, pretrained_path="D:/models/roberta-base", out_features=None):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_path)
        if out_features is None:
            out_features = self.model.pooler.dense.out_features
        self.probing_layer = LinearGroupNJ(
            self.model.pooler.dense.in_features, out_features
        )

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs for the transformer model.
            attention_mask (torch.Tensor, optional): Attention mask for the input tokens. Defaults to None.

        Returns:
            torch.Tensor: Output of the probing layer applied to the transformer's last hidden state.
        """
        with torch.no_grad():
            backbone_outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        return self.probing_layer(backbone_outputs.last_hidden_state)

    def kl_divergence(self):
        """
        Computes the KL divergence for the probing layer.

        Returns:
            torch.Tensor: KL divergence value for the Bayesian probing layer.
        """
        return self.probing_layer.kl_divergence()

    def train(self, *args, **kwargs):
        """
        Puts the model into training mode.

        Notes:
            Sets the probing layer to non-deterministic mode for variational training.

        Args:
            *args: Positional arguments for the training method.
            **kwargs: Keyword arguments for the training method.

        Returns:
            nn.Module: The model in training mode.
        """
        self.probing_layer.deterministic = False
        return super().train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        """
        Puts the model into evaluation mode.

        Notes:
            Sets the probing layer to deterministic mode for evaluation.

        Args:
            *args: Positional arguments for the evaluation method.
            **kwargs: Keyword arguments for the evaluation method.

        Returns:
            nn.Module: The model in evaluation mode.
        """
        self.probing_layer.deterministic = True
        return super().eval(*args, **kwargs)
