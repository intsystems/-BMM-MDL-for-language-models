from tqdm.auto import tqdm
import warnings
from BayesianLayers import *
import numpy as np

class Trainer:
    def __init__(
        self, 
        model=None,
        train_config=None
    ):
        """
        Initialize the Trainer object.

        Parameters
        ----------
        model : nn.Module
            Model to be trained.
        train_config : dict
            Configuration for training. Must be provided either here or in .train() method
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
        ignore_index=None,
        evaluate_every=-1,  # -1 for no evaluation
        KL_weight=1,
    ):

        """
        Train the model using the provided data loaders and training configuration.

        Parameters
        ----------
        train_loader : DataLoader, optional
            DataLoader for training data. Must be provided if not specified in train_config.
        val_loader : DataLoader, optional
            DataLoader for validation data. If not provided, evaluation will be skipped.
        model : nn.Module, optional
            Model to be trained. If not provided, the model must be set during initialization.
        n_epochs : int, optional
            Number of training epochs. If not provided, defaults to the value in train_config or 1.
        loss_function : str or callable, optional
            Loss function for training. Defaults to the value in train_config or 'crossentropy' if not specified.
        optimizer : torch.optim.Optimizer or str, optional
            Optimizer for training. If not provided, defaults to 'Adam'.
        variational : bool, optional
            Whether to use variational training. Defaults to the value in train_config or False if not specified.
        lr : float, optional
            Learning rate for the optimizer. Defaults to 1e-3 if not specified.
        evaluate_every : int, optional
            Frequency of evaluation during training. Defaults to -1 (no evaluation).
        ignore_index : int, optional
            Index to ignore in the loss function. Defaults defaults to the value in train_config or None.
        KL_weight : float, optional
            Weight for the KL term in the loss function. Defaults to 1 if not specified.

        Returns
        -------
        nn.Module
            The trained model.

        Raises
        ------
        ValueError
            If no model or train_loader is provided.
            If an unknown optimizer or loss function is specified.

        Warnings
        --------
        UserWarning
            If no learning rate, optimizer, loss_function, n_epochs or val_loader provided.
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
            warnings.warn("No loss function provided. Defaulting to crossentropy", UserWarning,)
            loss_function = "crossentropy"
            self.train_config["loss_function"] = loss_function

        if ignore_index is None:
            ignore_index = self.train_config.get("ignore_index", None)
        else:
            self.train_config["ignore_index"] = ignore_index

        if loss_function == "crossentropy":
            loss_function = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
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
            warnings.warn("No evaluate_every provided. Defaulting to -1 (no evaluation)", UserWarning)
            evaluate_every = -1

        self.KL_weight = KL_weight

        losses = []
        metrics = []

        if variational:
            eval_metrics = self.train_config.get("eval_metrics", [])
            self.train_config["eval_metrics"] = eval_metrics + ["description_length"]

        for epoch in tqdm(range(n_epochs), desc="training epoch"):
            losses_epoch = self._train_epoch(train_loader, loss_function, optimizer)
            losses.extend(losses_epoch)

            if evaluate_every > 0 and (epoch + 1) % evaluate_every == 0:
                epoch_metrics = self.evaluate(val_loader)
                metrics.append(epoch_metrics)

        self._metrics = metrics
        self._losses = losses

        return self.model

    def _train_epoch(
        self, 
        train_loader, 
        loss_function, 
        optimizer
    ):
        """
        Train the model for one epoch.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The data loader for the training data.
        loss_function: torch.nn.Module
            The loss function to use for training.
        optimizer: torch.optim.Optimizer
            The optimizer to use for training.

        Returns
        -------
        list
            A list of losses for each batch in the epoch.
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
            
            loss = loss_function(output.view(-1, output.shape[-1]), batch[-1].to(device).view(-1))

            if variational:
                kl_loss = sum([m.kl_divergence() for m in bayes_modules])
                loss = loss + kl_loss * self.KL_weight

            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())

        return losses

    def _forward(self, batch, device="cuda"):
        """
        Performs a forward pass of the model using the input batch.

        Parameters
        ----------
        batch : tuple
            A tuple containing input_ids and attention_mask tensors.
        device : str, optional
            The device to run the forward pass on, defaults to "cuda".

        Returns
        -------
        torch.Tensor
            The model predictions for the given inputs.
        """
        input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
        preds = self.model(input_ids, attention_mask)

        return preds
    
    def _calulate_metrics(self, output, batch, device="cuda"):
        """
        Calculates the given metrics for the given output and batch.

        Parameters
        ----------
        output : torch.Tensor
            The model predictions for the given inputs.
        batch : tuple
            A tuple containing input_ids, attention_mask and labels tensors.
        device : str, optional
            The device to run the forward pass on, defaults to "cuda".

        Returns
        -------
        dict
            A dictionary containing the calculated metrics.
        """
        metrics_to_calulate = self.train_config.get("eval_metrics", [])
        metrics = {}

        for m in metrics_to_calulate:
            if m == "description_length":
                ce_part = nn.CrossEntropyLoss(ignore_index=self.train_config["ignore_index"], reduction="sum")(output.view(-1, output.shape[-1]), batch[-1].to(device).view(-1)).detach().cpu().numpy()
                kl_part = self.model.kl_divergence().detach().cpu().numpy()
                metrics["cross_entropy_loss"] = ce_part
                metrics["KL_divergence"] = kl_part
                metrics[m] = ce_part + kl_part
            elif m == "accuracy":
                metrics[m] = (output.argmax(dim=-1) == batch[-1].to(device)).float().mean().detach().cpu().numpy()
            elif m == "loss" or m == "cross_entropy_loss":
                loss_function = self.train_config.get("loss_function", None)
                if loss_function is not None and "description_length" not in metrics_to_calulate:
                    if isinstance(loss_function, str):
                        if loss_function == "crossentropy":
                            loss_function = nn.CrossEntropyLoss(ignore_index=self.train_config["ignore_index"], reduction="sum")
                        elif loss_function == "mse":
                            loss_function = nn.MSELoss(reduction="sum")
                        else:
                            raise ValueError(f"Unknown loss function: {loss_function}")
                    loss = loss_function(output.view(-1, output.shape[-1]), batch[-1].to(device).view(-1))
                    metrics[m] = loss.detach().cpu().numpy()
            else:
                warnings.warn(f"Unknown metric: {m}", UserWarning)
        
        return metrics

    def evaluate(
        self,
        val_loader=None
    ):  
        """
        Evaluates the model on the given validation data.

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader, optional
            The validation data loader. If not provided, evaluation is skipped and empty validation metrics are returned.

        Returns
        -------
        dict
            A dictionary containing the mean validation metrics.
            
        Warnings
        --------
        UserWarning
            If no validation data loader is provided and an empty dictionary is returned.
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

                    metrics.append(batch_metrics)

                metrics_mean = {}
                for metric in metrics[0].keys():
                    metrics_mean[metric] = np.mean([m[metric] for m in metrics])
                    if metric == "description_length":
                        metrics_mean["cross_entropy_loss"] = np.mean([m["cross_entropy_loss"] for m in metrics])
                        metrics_mean["KL_divergence"] = np.mean([m["KL_divergence"] for m in metrics])

                return metrics_mean

        else:
            warnings.warn("No validation data provided, return empty val metrics", UserWarning)
            return {}
