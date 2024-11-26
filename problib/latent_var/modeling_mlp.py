import torch
from torch import nn
from transformers import PretrainedConfig
from .samplers import get_sampler
from .base import BaseModel


class MLPConfig(PretrainedConfig):
    """Configuration class for the MLPClassifier model.

    Attributes:
        model_type (str): Type of the model (default: "mlp_classifier").
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Dimensionality of the hidden layers.
        output_dim (int): Number of output classes.
        num_layers (int): Number of hidden layers in the MLP.
        sampler_type (str): Type of sampler to use.
        D (int): Dimensionality of the input features (alias for input_dim).
        K (int): Number of samples or features to consider.
    """

    model_type = "mlp_classifier"

    def __init__(
        self,
        K: int,
        input_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 10,
        num_layers: int = 2,
        sampler_type: str = "poisson",
        **kwargs  # Accept additional keyword arguments for PretrainedConfig
    ):
        """Initialize the MLPConfig object.

        Args:
            K (int): Number of samples or features to consider.
            input_dim (int, optional): Dimensionality of the input features. Defaults to 768.
            hidden_dim (int, optional): Dimensionality of the hidden layers. Defaults to 256.
            output_dim (int, optional): Number of output classes. Defaults to 10.
            num_layers (int, optional): Number of hidden layers in the MLP. Defaults to 2.
            sampler_type (str, optional): Type of sampler to use. Defaults to "poisson".
            **kwargs: Additional keyword arguments for PretrainedConfig initialization.
        """
        super().__init__(**kwargs)  # Initialize base configuration

        # Set configuration parameters as attributes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.sampler_type = sampler_type
        self.D = input_dim  # Dimensionality of input features
        self.K = K  # Number of samples or features to consider


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron Classifier model.

    This model consists of multiple linear layers followed by ReLU activations.

    Args:
        nn.Module: Base class for all neural network modules in PyTorch.

    Attributes:
        layers (nn.Sequential): Sequential container for the MLP layers.
    """

    def __init__(self, config: MLPConfig):
        """Initialize the MLPClassifier model.

        Args:
            config (MLPConfig): Configuration object containing model parameters.
        """
        super().__init__()

        layers = []
        input_dim = config.input_dim

        # Create layers based on the configuration
        for _ in range(config.num_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))  # Add linear layer
            layers.append(nn.ReLU())  # Add ReLU activation
            input_dim = config.hidden_dim  # Update input dimension for next layer

        layers.append(nn.Linear(config.hidden_dim, config.output_dim))  # Output layer

        # Store constructed layers in a sequential container as an attribute
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLPClassifier model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits with shape (batch_size, output_dim).
        """
        return self.layers(x)


class ProbingModel(BaseModel):
    """Probing Model for analysis.

    This model integrates an MLP classifier with a sampling mechanism.

    Args:
        BaseModel: Base class for models in this library.

    Attributes:
        name (str): Name identifier for the probing model.
        config (MLPConfig): Configuration object containing model parameters.
        sampler: Sampler instance used for generating masks.
        classifier (MLPClassifier): Instance of the MLPClassifier used for predictions.
    """

    name = "probing_model"

    def __init__(self, config: MLPConfig):
        """Initialize the ProbingModel.

        Args:
            config (MLPConfig): Configuration object containing model parameters.
        """
        super().__init__()

        # Store configuration as an attribute
        self.config = config

        # Initialize sampler based on configuration as an attribute
        self.sampler = get_sampler(config.sampler_type, config.D, config.K)

        # Initialize classifier using provided configuration as an attribute
        self.classifier = MLPClassifier(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ProbingModel.

        Args:
             x (torch.Tensor): Input tensor with shape (batch_size, D).

        Returns:
             torch.Tensor: Output logits after applying sampling and classification.

        The method samples a mask from the sampler and applies it to the input tensor before passing it through
        the classifier. The resulting logits are returned as output.
        """

        # Sample a mask based on batch size from sampler
        mask = self.sampler.sample(x.size(0))

        # Apply mask to input tensor
        masked_x = x * mask

        # Get logits from classifier based on masked inputs
        logits = self.classifier(masked_x)

        return logits
