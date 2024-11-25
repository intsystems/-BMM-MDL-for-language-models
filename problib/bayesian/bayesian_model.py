import math
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from base import BaseModel

class MLPConfig(PretrainedConfig):
    """Configuration class for the MLPClassifier model.

    Attributes:
        model_type (str): Type of the model (default: "mlp_classifier").
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Dimensionality of the hidden layers.
        output_dim (int): Number of output classes.
        num_layers (int): Number of hidden layers in the MLP.
        dropout (float): Dropout rate for regularization.
        vocab_size (int): Size of the vocabulary for embedding.
    """
    model_type = "mlp_classifier"

    def __init__(
        self,
        K: int,
        input_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
        vocab_size: int = 30522,
        **kwargs  # Additional keyword arguments for PretrainedConfig
    ):
        """Initialize the MLPConfig object.

        Args:
            K (int): Number of samples or features to consider.
            input_dim (int, optional): Dimensionality of the input features. Defaults to 768.
            hidden_dim (int, optional): Dimensionality of the hidden layers. Defaults to 256.
            output_dim (int, optional): Number of output classes. Defaults to 3.
            num_layers (int, optional): Number of hidden layers in the MLP. Defaults to 2.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.1.
            vocab_size (int, optional): Size of the vocabulary for embedding. Defaults to 30522.
            **kwargs: Additional keyword arguments for PretrainedConfig initialization.
        """
        super().__init__(**kwargs)

        # Set configuration parameters as attributes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.D = input_dim  # Dimensionality of input features
        self.K = K          # Number of samples or features to consider
        self.dropout = dropout  # Dropout rate for regularization
        self.vocab_size = vocab_size  # Vocabulary size

class MLP(BaseModel):
    """Multi-layer Perceptron model.

    This model consists of multiple linear layers followed by ReLU activations.

    Attributes:
        name (str): Name identifier for the model ("mlp").
        mlp (nn.Sequential): Sequential container for MLP layers.
        out (nn.Linear): Output layer mapping from hidden size to output dimension.
        dropout (nn.Dropout): Dropout layer for regularization during training.
    """

    name = "mlp"

    def __init__(self, task: str, config: MLPConfig):
        """Initialize the model.

        Args:
            task (str): The task to perform (e.g., "dep_label").
            config (MLPConfig): The configuration for the model.
        """
        super().__init__()

        # Build embeddings if representation is onehot or random
        if self.representation in ["onehot", "random"]:
            self.build_embeddings(config.vocab_size, config.hidden_dim)

        # Build MLP structure and output layer
        self.mlp = self.build_mlp()
        
        # Output layer mapping from final hidden size to output dimension
        self.out = nn.Linear(self.final_hidden_size, config.output_dim)
        
        # Dropout layer for regularization during training
        self.dropout = nn.Dropout(config.dropout)

        # Loss function criterion
        self.criterion = nn.CrossEntropyLoss()

    def build_embeddings(self, n_words: int, embedding_size: int):
        """Build the embeddings for the model.

        Args:
            n_words (int): The number of words in vocabulary.
            embedding_size (int): The size of each embedding vector.
        
        This method initializes an embedding layer based on task requirements and representation type.
        """
        
        if self.task == "dep_label":
            self.embedding_size = int(embedding_size / 2) * 2
            self.embedding = nn.Embedding(n_words, int(embedding_size / 2))
        else:
            self.embedding = nn.Embedding(n_words, embedding_size)

        if self.representation == "random":
            self.embedding.weight.requires_grad = False

    def build_mlp(self):
        """Builds the MLP architecture based on configuration parameters.

        Returns:
             nn.Sequential: A sequential container with linear layers and activations configured according to num_layers and dropout settings.
        """
        if self.num_layers == 0:
            self.final_hidden_size = self.embedding_size
            return nn.Identity()

        src_size = self.embedding_size
        tgt_size = self.hidden_size
        mlp_layers = []
         
        # Create multiple layers based on configuration
        for _ in range(self.num_layers):
            mlp_layers.append(nn.Linear(src_size, tgt_size))      # Linear layer
            mlp_layers.append(nn.ReLU())                           # ReLU activation
            mlp_layers.append(nn.Dropout(self.dropout))           # Dropout layer
            
            src_size, tgt_size = tgt_size, int(tgt_size / 2)     # Update sizes
        
        self.final_hidden_size = src_size  # Final hidden size after all layers
        return nn.Sequential(*mlp_layers)   # Return constructed MLP

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor with shape [batch_size, ...].

        Returns:
            torch.Tensor: The output logits from the model.
        """
        
        if self.representation in ["onehot", "random"]:
            x = self.get_embeddings(x)  # Get embeddings based on representation type

        x_emb = self.dropout(x)       # Apply dropout on embeddings
        x_mlp_out = self.mlp(x_emb)   # Pass through MLP layers
         
        logits = self.out(x_mlp_out)  # Get final logits from output layer
         
        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for input data.

        Args:
            x (torch.Tensor): Input tensor containing indices.

        Returns:
            torch.Tensor: The resulting embeddings after processing through embedding layer.
        """
        
        x_emb = self.embedding(x)      # Get embeddings from embedding layer
        
        if len(x.shape) > 1:
            x_emb = x_emb.reshape(x.shape[0], -1)  # Reshape if necessary
        
        return x_emb

    def train_batch(self, data: torch.Tensor, target: torch.Tensor, optimizer):
        """Train the model on a single batch.

        Args:
            data (torch.Tensor): Input tensor containing data samples.
            target (torch.Tensor): Target tensor containing labels corresponding to data samples.
            optimizer (torch.optim.Optimizer): Optimizer instance used for parameter updates.

        Returns:
            float: The computed loss value after training on this batch.
        """
        
        optimizer.zero_grad()          # Clear previous gradients
        
        mlp_out = self(data)           # Forward pass through model
        
        loss_value = self.criterion(mlp_out, target)   # Compute loss based on predictions and targets
        
        loss_value.backward()           # Backpropagation step
        
        optimizer.step()                # Update parameters using optimizer

        return loss_value.item() / math.log(2)  # Return normalized loss value

    def eval_batch(self, data: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
        """Evaluate performance on a single batch.

        Args:
            data (torch.Tensor): Input tensor containing data samples for evaluation.
            target (torch.Tensor): Target tensor containing labels corresponding to data samples.

        Returns:
            tuple: A tuple containing total loss and accuracy over this batch.
        """
          
        mlp_out = self(data)           # Forward pass through model
        
        loss_value = self.criterion(mlp_out, target) / math.log(2)  # Compute loss value
        
        accuracy_value = (mlp_out.argmax(dim=-1) == target).float().detach().sum()   # Calculate accuracy
        
        total_loss_value = loss_value.item() * data.shape[0]   # Scale loss by batch size

        return total_loss_value, accuracy_value   # Return total loss and accuracy

    @staticmethod
    def get_norm() -> torch.Tensor:
        """Get norm value associated with this model.

        Returns:
            torch.Tensor: A tensor representing norm value; placeholder implementation here returns zero tensor.
        """
          
        return torch.Tensor([0])   # Placeholder implementation returning zero tensor

    def get_args(self) -> Dict[str, Union[int, str]]:
        """Get arguments relevant to this model configuration.

        Returns:
            dict: A dictionary containing relevant configuration arguments as key-value pairs.
        """
          
        return {
            "nlayers": self.num_layers,
            "hidden_size": self.hidden_size,
            "embedding_size": self.embedding_size,
            "dropout": self.dropout,
            "n_classes": self.output_dim,
            "representation": getattr(self, 'representation', None),  # Use getattr to avoid error if attribute doesn't exist
            "n_words": getattr(self, 'vocab_size', None),           # Use getattr similarly here 
            "task": getattr(self, 'task', None),                    # Use getattr similarly here 
        }

    @staticmethod
    def print_param_names() -> list:
        """Print parameter names relevant to this model configuration.

        Returns:
            list: A list of parameter names as strings.
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

    def print_params(self) -> list:
        """Print parameters relevant to this model configuration.

        Returns:
            list: A list containing current parameter values as stored in attributes.
        """
          
        return [
            self.num_layers,
            self.hidden_dim,
            self.embedding_size,
            self.dropout,
            self.output_dim,
            getattr(self, 'representation', None),  # Use getattr here too 
            getattr(self, 'vocab_size', None),      # Use getattr here too 
        ]