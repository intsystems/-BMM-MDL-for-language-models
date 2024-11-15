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
        if model is not None:
            self.model = model
        if self.model is None:
            raise ValueError("No model provided")

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
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif opt_name == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        if n_epochs is None:
            n_epochs = self.train_config.get("n_epochs", 1)
        else:
            self.train_config["n_epochs"] = n_epochs

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

        if loss_function == "crossentropy":
            loss_function = nn.CrossEntropyLoss()
        elif loss_function == "mse":
            loss_function = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

        if train_loader is None:
            train_loader = self.train_config.get("train_loader", None)

        if variational is not None:
            self.train_config["variational"] = variational

        if lr is not None:
            self.train_config["lr"] = lr
        elif "lr" in self.train_config:
            lr = self.train_config["lr"]
        else:
            warnings.warn("No learning rate provided. Defaulting to 1e-3", UserWarning)
            lr = 1e-3

        if evaluate_every is not None:
            self.train_config["evaluate_every"] = evaluate_every
        elif "evaluate_every" in self.train_config:
            evaluate_every = self.train_config["evaluate_every"]
        else:
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

    def _train_epoch(
        self, 
        train_loader, 
        loss_function, 
        optimizer
    ):
        losses = []
        device = next(self.model.parameters()).device

        variational = self.train_config.get("variational", False)

        if variational:
            bayes_modules = list(get_kl_modules(self.model))

        for batch in tqdm(train_loader, leave=False, desc="training batch"):
            optimizer.zero_grad()
            self.model.train()

            output = self._forward(batch, device=device)

            if "last_hidden_state" in output and isinstance(loss_function, nn.Module):
                output = output["last_hidden_state"]

            loss = loss_function(output, batch[-1])

            if variational:
                kl_loss = sum([m.kl_divergence() for m in bayes_modules])
                loss = loss + kl_loss

            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())

        return losses

    def _forward(self, batch, device="cuda"):
        input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
        preds = self.model(input_ids, attention_mask)

        return preds
    
    def _calulate_metrics(self, outputs, batch):
        metrics_to_calulate = self.train_config.get("eval_metrics", [])
        metrics = {}

        if "last_hidden_state" in outputs:
            pred = outputs["last_hidden_state"]
        else:
            pred = outputs

        for m in metrics_to_calulate:
            if m == "description_length":
                metrics[m] = nn.CrossEntropyLoss(reduction="mean")(pred, batch[0]).detach().cpu().numpy() + self.model.kl_divergence()
            elif m == "accuracy":
                metrics[m] = (pred.argmax(dim=1) == batch[-1]).mean().detach().cpu().numpy()
            else:
                warnings.warn(f"Unknown metric: {m}", UserWarning)
        
        return metrics

    def evaluate(
        self,
        val_loader=None,
        loss_function=None,
    ):  
        if val_loader is not None:
            with torch.no_grad():
                self.model.eval()
                device = next(self.model.parameters()).device

                metrics = []
                for batch in tqdm(val_loader, leave=False, desc="training batch"):
                    self.model.train()

                    output = self._forward(batch, device=device)
                    
                    self._calulate_metrics(output, batch)
                    batch_metrics = self._calulate_metrics(output, batch)

                    if not self.train_config.get("variational", False) and loss_function is not None:
                        if "last_hidden_state" in output and isinstance(loss_function, nn.Module):
                            output = output["last_hidden_state"]
                        loss = loss_function(output, batch[-1])
                        batch_metrics["loss"] = loss

                    metrics.append(batch_metrics)

                metrics_mean = {}
                for metric in self.train_config.get("eval_metrics", []):
                    metrics_mean[metric] = np.mean([m[metric] for m in metrics])

        else:
            warnings.warn("No validation data provided, return empty val metrics", UserWarning)
            return {}
