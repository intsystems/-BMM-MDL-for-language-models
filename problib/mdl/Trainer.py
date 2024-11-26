from tqdm.auto import tqdm
import warnings
from BayesianLayers import *
import numpy as np


class Trainer:
    """
    Trainer class for training models with various configurations and handling evaluation.

    Args:
        model (nn.Module, optional): The model to be trained. Defaults to None.
        train_config (dict, optional): Configuration dictionary for training. Defaults to None.

    Attributes:
        model (nn.Module): The model being trained.
        train_config (dict): Configuration settings for the training process.
    """

    def __init__(self, model=None, train_config=None):
        """
        Initialize the Trainer object.

        Args:
            model (nn.Module, optional): The model to train. Defaults to None.
            train_config (dict, optional): Training configuration dictionary. Defaults to None.
        """
        self.model = model
        self.train_config = train_config if train_config is not None else {}

    def train(
        self,
        train_loader=None,
        val_loader=None,
        model=None,
        n_epochs=None,
        loss_function=None,
        optimizer=None,
        variational=None,
        lr=None,
        evaluate_every=-1,  # -1 for no evaluation
    ):
        """
        Train the model using the provided configuration and data loaders.

        Args:
            train_loader (DataLoader, optional): DataLoader for the training data. Defaults to None.
            val_loader (DataLoader, optional): DataLoader for validation data. Defaults to None.
            model (nn.Module, optional): Model to train. Defaults to None.
            n_epochs (int, optional): Number of training epochs. Defaults to None.
            loss_function (str or callable, optional): Loss function for training. Defaults to None.
            optimizer (torch.optim.Optimizer or str, optional): Optimizer to use. Defaults to None.
            variational (bool, optional): If True, uses variational training. Defaults to None.
            lr (float, optional): Learning rate for the optimizer. Defaults to None.
            evaluate_every (int, optional): Evaluation frequency (in epochs). Defaults to -1.

        Returns:
            nn.Module: The trained model.

        Raises:
            ValueError: If no model or training loader is provided.
            ValueError: If an unknown optimizer or loss function is specified.
        """
        if model is not None:
            self.model = model
        if self.model is None:
            raise ValueError("No model provided")

        if lr is not None:
            self.train_config["lr"] = lr
        elif "lr" in self.train_config:
            lr = self.train_config["lr"]
        else:
            warnings.warn("No learning rate provided. Defaulting to 1e-3", UserWarning)
            lr = 1e-3

        if optimizer is not None:
            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer = optimizer(model.parameters(), lr=lr)
            else:
                self.train_config["optimizer"] = optimizer
        elif "optimizer" in self.train_config:
            opt_name = self.train_config["optimizer"]
        else:
            warnings.warn("No optimizer provided. Defaulting to Adam", UserWarning)
            opt_name = "Adam"

        if opt_name == "Adam":
            optimizer = torch.optim.Adam(self.model.probing_layer.parameters(), lr=lr)
        elif opt_name == "SGD":
            optimizer = torch.optim.SGD(self.model.probing_layer.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        if n_epochs is not None:
            self.train_config["n_epochs"] = n_epochs
        elif "n_epochs" in self.train_config:
            n_epochs = self.train_config["n_epochs"]
        else:
            n_epochs = 1
            self.train_config["n_epochs"] = n_epochs
            warnings.warn("No n_epochs provided. Defaulting to 1", UserWarning)

        if loss_function is not None:
            if not isinstance(loss_function, str):
                pass
            else:
                self.train_config["loss_function"] = loss_function
        elif "loss_function" in self.train_config:
            loss_function = self.train_config["loss_function"]
        else:
            warnings.warn(
                "No loss function provided. Defaulting to crossentropy",
                UserWarning,
            )
            loss_function = "crossentropy"
            self.train_config["loss_function"] = loss_function

        if loss_function == "crossentropy":
            loss_function = nn.CrossEntropyLoss()
        elif loss_function == "mse":
            loss_function = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

        if train_loader is None:
            train_loader = self.train_config.get("train_loader", None)
            if train_loader is None:
                raise ValueError("No train_loader provided")

        if val_loader is None:
            val_loader = self.train_config.get("val_loader", None)
            if val_loader is None:
                warnings.warn("No val_loader provided", UserWarning)

        if variational is not None:
            self.train_config["variational"] = variational
        else:
            variational = self.train_config.get("variational", False)

        if evaluate_every is not None:
            self.train_config["evaluate_every"] = evaluate_every
        elif "evaluate_every" in self.train_config:
            evaluate_every = self.train_config["evaluate_every"]
        else:
            warnings.warn(
                "No evaluate_every provided. Defaulting to -1 (no evaluation)",
                UserWarning,
            )
            evaluate_every = -1

        losses = []
        metrics = []

        if variational:
            eval_metrics = self.train_config.get("eval_metrics", [])
            self.train_config["eval_metrics"] = eval_metrics + ["description_length"]

        for epoch in tqdm(range(n_epochs), desc="training epoch"):
            losses_epoch = self._train_epoch(train_loader, loss_function, optimizer)
            losses.extend(losses_epoch)

            if evaluate_every > 0 and (epoch + 1) % evaluate_every == 0:
                epoch_metrics = self.evaluate(val_loader, loss_function)
                metrics.append(epoch_metrics)

        self._metrics = metrics
        self._losses = losses

        return self.model

    def _train_epoch(self, train_loader, loss_function, optimizer):
        """
        Perform one epoch of training.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            loss_function (torch.nn.Module): Loss function to use for training.
            optimizer (torch.optim.Optimizer): Optimizer for training.

        Returns:
            list: A list of batch losses for the epoch.
        """
        losses = []
        device = next(self.model.parameters()).device

        variational = self.train_config.get("variational", False)

        if variational:
            bayes_modules = list(get_kl_modules(self.model))

        for batch in tqdm(train_loader, leave=False, desc="training batch"):
            optimizer.zero_grad()
            self.model.train()

            output = self._forward(batch, device=device)

            loss = loss_function(
                output.view(-1, output.shape[-1]), batch[-1].to(device).view(-1)
            )

            if variational:
                kl_loss = sum([m.kl_divergence() for m in bayes_modules])
                loss = loss + kl_loss

            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())

        return losses

    def _forward(self, batch, device="cuda"):
        """
        Perform a forward pass on the given batch.

        Args:
            batch (tuple): Input batch containing input data and labels.
            device (str, optional): Device for computation. Defaults to "cuda".

        Returns:
            torch.Tensor: Model predictions.
        """
        input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
        preds = self.model(input_ids, attention_mask)

        return preds

    def _calulate_metrics(self, output, batch, device="cuda"):
        """
        Calculate metrics for a batch.

        Args:
            output (torch.Tensor): Model predictions.
            batch (tuple): Input batch with labels.
            device (str, optional): Device for computation. Defaults to "cuda".

        Returns:
            dict: Calculated metrics.
        """
        metrics_to_calulate = self.train_config.get("eval_metrics", [])
        metrics = {}

        for m in metrics_to_calulate:
            if m == "description_length":
                ce_part = (
                    nn.CrossEntropyLoss()(
                        output.view(-1, output.shape[-1]), batch[-1].to(device).view(-1)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                kl_part = self.model.kl_divergence().detach().cpu().numpy()
                metrics[m] = ce_part + kl_part
            elif m == "accuracy":
                metrics[m] = (
                    (output.argmax(dim=-1) == batch[-1].to(device))
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                warnings.warn(f"Unknown metric: {m}", UserWarning)

        return metrics

    def evaluate(
        self,
        val_loader=None,
        loss_function=None,
    ):
        """
        Evaluate the model on the validation dataset.

        Args:
            val_loader (DataLoader, optional): Validation DataLoader. Defaults to None.
            loss_function (callable or str, optional): Loss function for evaluation. Defaults to None.

        Returns:
            dict: Average metrics over the validation dataset.
        """
        if val_loader is not None:
            with torch.no_grad():
                self.model.eval()
                device = next(self.model.parameters()).device

                metrics = []
                for batch in tqdm(val_loader, leave=False, desc="eval batch"):
                    self.model.train()

                    output = self._forward(batch, device=device)

                    batch_metrics = self._calulate_metrics(output, batch, device)

                    if loss_function is None:
                        loss_function = self.train_config.get("loss_function", None)
                    if (
                        not self.train_config.get("variational", False)
                        and loss_function is not None
                    ):
                        if isinstance(loss_function, str):
                            if loss_function == "crossentropy":
                                loss_function = nn.CrossEntropyLoss()
                            elif loss_function == "mse":
                                loss_function = nn.MSELoss()
                            else:
                                raise ValueError(
                                    f"Unknown loss function: {loss_function}"
                                )
                        loss = loss_function(
                            output.view(-1, output.shape[-1]),
                            batch[-1].to(device).view(-1),
                        )
                        batch_metrics["loss"] = loss

                    metrics.append(batch_metrics)

                metrics_mean = {}
                for metric in self.train_config.get("eval_metrics", []):
                    metrics_mean[metric] = np.mean([m[metric] for m in metrics])

                return metrics_mean

        else:
            warnings.warn(
                "No validation data provided, return empty val metrics", UserWarning
            )
            return {}
