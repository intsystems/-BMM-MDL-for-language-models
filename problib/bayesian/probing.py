from ..base import BaseModel
from bayesian_model import MLP
import torch
from transformers import AutoModel, AutoTokenizer
import os
import sys
from tqdm import tqdm
import torch.optim as optim
from utils import TrainInfo

# Adjust the system path to include the parent directory
sys.path.insert(1, os.path.join(sys.path[0], ".."))


class BayesianProbingModel(BaseModel):
    """
    Bayesian Probing Model that utilizes a multi-layer perceptron (MLP) for classification tasks.

    Args:
        BaseModel: Base class for models in this library.

    Attributes:
        model (MLP): The MLP model used for classification.
        name (str): Identifier for the model ("mlp").
    """

    name = "mlp"

    def __init__(
        self,
        embedding_size: int = 512,
        n_classes: int = 2,
        hidden_size: int = 512,
        n_layers: int = 10,
        dropout: float = 0.1,
        representation: str = None,
        n_words: int = 10,
        device: str = "cuda",
    ):
        """
        Initialize the Bayesian Probing Model.

        Args:
            embedding_size (int): The size of the embedding layer.
            n_classes (int): The number of output classes.
            hidden_size (int): The size of the hidden layers.
            n_layers (int): The number of layers in the MLP.
            dropout (float): The dropout rate for regularization.
            representation (str, optional): The type of representation to use. Defaults to None.
            n_words (int): The number of words in the vocabulary.
            device (str): The device to run the model on (e.g., "cuda" or "cpu").
        """
        super().__init__()

        # Initialize the MLP model with specified parameters
        self.model = MLP(
            embedding_size=embedding_size,
            n_classes=n_classes,
            hidden_size=hidden_size,
            nlayers=n_layers,
            dropout=dropout,
            representation=representation,
            n_words=n_words,
        )

        # Move the model to the specified device
        self.model.to(device)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): The input IDs of the tokens.
            attention_mask (Optional[torch.Tensor]): The attention mask for the tokens. Defaults to None.

        Returns:
            torch.Tensor: The output logits from the model.
        """
        if attention_mask is not None:
            return self.model(input_ids=input_ids, attention_mask=attention_mask)

        return self.model(input_ids=input_ids)

    def evaluate(
        self, evalloader: torch.utils.data.DataLoader, model
    ) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.

        Args:
            evalloader (DataLoader): The evaluation dataset loader.
            model (BayesianProbingModel): The model to evaluate.

        Returns:
            dict: A dictionary containing the evaluation loss and accuracy.
        """
        model.eval()  # Set the model to evaluation mode
        dev_loss, dev_acc = 0.0, 0.0

        with torch.no_grad():  # Disable gradient calculation for evaluation
            for x, y in evalloader:
                loss, acc = model.eval_batch(x, y)  # Evaluate batch
                dev_loss += loss  # Accumulate loss
                dev_acc += acc  # Accumulate accuracy

            n_instances = len(
                evalloader.dataset
            )  # Total number of instances in evaluation dataset

            result = {
                "loss": dev_loss / n_instances,  # Average loss over all instances
                "acc": dev_acc / n_instances,  # Average accuracy over all instances
            }

        model.train()  # Set back to training mode
        return result

    def train_epoch(
        self,
        trainloader: torch.utils.data.DataLoader,
        devloader: torch.utils.data.DataLoader,
        model,
        optimizer: optim.Optimizer,
        train_info: TrainInfo,
    ):
        """
        Train the model for one epoch.

        Args:
            trainloader (DataLoader): The training dataset loader.
            devloader (DataLoader): The development dataset loader.
            model (BayesianProbingModel): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            train_info (TrainInfo): Object containing training information and progress tracking.
        """

        for x, y in trainloader:
            loss = model.train_batch(x, y, optimizer)  # Train on current batch
            train_info.new_batch(loss)  # Update training info with new batch loss

            if train_info.eval:
                dev_results = self.evaluate(
                    devloader, model
                )  # Evaluate on development set

                if train_info.is_best(dev_results):
                    model.set_best()  # Mark current model as best if results improve

                elif train_info.finish:
                    train_info.print_progress(
                        dev_results
                    )  # Print progress if training is finished
                    return

                train_info.print_progress(
                    dev_results
                )  # Print current evaluation results

    def train(
        self,
        trainloader: torch.utils.data.DataLoader,
        devloader: torch.utils.data.DataLoader,
        model,
        eval_batches: int,
        wait_iterations: int,
    ):
        """
        Train the model over multiple epochs until completion or stopping criteria are met.

        Args:
            trainloader (DataLoader): The training dataset loader.
            devloader (DataLoader): The development dataset loader.
            model (BayesianProbingModel): The model to train.
            eval_batches (int): Number of batches to evaluate after each epoch.
            wait_iterations (int): Number of iterations to wait before stopping early.

        This method manages the overall training loop and progress reporting using tqdm for visualization.
        """

        optimizer = optim.AdamW(model.parameters())  # Initialize optimizer

        with tqdm(total=wait_iterations) as pbar:
            train_info = TrainInfo(
                pbar, wait_iterations, eval_batches
            )  # Initialize training info tracker

            while not train_info.finish:
                self.train_epoch(
                    trainloader, devloader, model, optimizer, train_info
                )  # Train for one epoch

        model.recover_best()  # Restore best-performing model after training completes
