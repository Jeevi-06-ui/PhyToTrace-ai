"""Confidence calibration utilities using temperature scaling."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim


class ModelWithTemperature(nn.Module):
    """Post-hoc temperature scaling calibration wrapper for PyTorch classification heads."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def calibrate(self, val_loader, device: torch.device, max_iter: int = 50) -> float:
        """Find optimal temperature on validation set via NLL."""
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss()
        
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                logits = self.model(x)
                logits_list.append(logits)
                labels_list.append(y.to(device))
                
        if not logits_list:
            return 1.0
            
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        optimal_t = float(self.temperature.item())
        return optimal_t

