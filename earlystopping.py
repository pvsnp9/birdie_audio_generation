import torch
import torch.nn as nn
import os 

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """
        Args:
            patience (int): Number of epochs to wait after the last improvement.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, model_dir: str, model_file_name: str) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, model_dir, model_file_name)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model, model_dir, model_file_name)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    @staticmethod
    def save_checkpoint(model: nn.Module, model_dir: str, model_file_name) -> None:
        """Saves the model when the validation loss decreases."""
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        torch.save(model.state_dict(), os.path.join(model_dir, model_file_name))
