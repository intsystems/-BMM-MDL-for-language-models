from tqdm.auto import tqdm
import warnings
from BayesianLayers import *


class Trainer:
    def __init__(
        self,
        eval_metrics=None,
        model=None
    ):
        self.eval_metrics = eval_metrics # TODO: add eval metrics
        self.model = model

    def train(
        self,
        model=None,
        train_loader=None,
        val_loader=None,
        n_epochs=1,
        loss_function=None,
        optimizer=None,
        lr=3e-4,
        evaluate_every=-1, # -1 for no evaluation
    ):  
        if model is not None:
            self.model = model
        if self.model is None:
            raise ValueError("No model provided")
        
        optimizer = optimizer(model.parameters(), lr=lr)

        losses = []
        metrics = []
        for epoch in tqdm(range(n_epochs), desc="training epoch"):
            losses_epoch = self._train_epoch(
                train_loader,
                loss_function,
                optimizer
            )
            losses.extend(losses_epoch)
            
            if evaluate_every > 0 and (epoch + 1) % evaluate_every == 0:
                epoch_metrics = self.evaluate(
                    val_loader,
                    loss_function
                )
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
        bayes_modules = list(get_kl_modules(self.model))

        for batch in tqdm(train_loader, leave=False, desc="training batch"):
            optimizer.zero_grad()
            self.model.train()
            
            output_dict = self._forward(batch, device=device)

            loss = loss_function(output_dict, batch)

            if len(bayes_modules) > 0:
                kl_loss = sum([m.kl_divergence() for m in bayes_modules])
                loss = loss + kl_loss

            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())
        
        return losses
    
    def _forward(
        self,
        batch,
        device="cuda"
    ):
        input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
        preds = self.model(input_ids, attention_mask)

        return preds
    
    def _evaluate(
        self,
        val_loader=None,
    ):
        # TODO: implement
        warnings.warn("evaluation not implemented! Returning empty metric dict")
        return {}