import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional


class EvidentialNetwork(nn.Module):
    """
    Evidential Deep Learning Network.
    Outputs Dirichlet distribution parameters (Alphas) instead of probabilities.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, dropout: float):
        super(EvidentialNetwork, self).__init__()

        # Layer 1
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Layer 2
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        # Output Layer (Outputs 'Evidence')
        self.output_layer = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)

        logits = self.output_layer(x)

        # Softplus ensures evidence is >= 0
        evidence = F.softplus(logits)

        # Alpha = Evidence + 1 (Dirichlet parameters)
        alpha = evidence + 1
        return alpha


def calculate_epistemic_uncertainty(alpha: torch.Tensor) -> float:
    """
    Calculates Model Uncertainty (Epistemic).
    Formula: K / Sum(alpha)
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    K = alpha.shape[1]
    uncertainty = K / S
    return float(uncertainty.item())


# ==============================================================================
# LOSS FUNCTIONS (Evidence Lower Bound)
# ==============================================================================

def kl_divergence(alpha: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    """Calculates KL Divergence between predicted Dirichlet and Uniform Dirichlet."""
    beta = torch.ones((1, num_classes)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def edl_loss(
        func: Optional[Callable],
        y: torch.Tensor,
        alpha: torch.Tensor,
        epoch_num: int,
        num_classes: int,
        annealing_step: int,
        device: torch.device,
        lamb: float = 1.0
) -> torch.Tensor:
    """
    EDL Loss: Sum of Squares Error + Variance + KL Divergence (Regularization).
    """
    y = y.to(device)
    alpha = alpha.to(device)
    y = y.view(-1)

    # One-hot encoding of the true class
    y_one_hot = F.one_hot(y, num_classes).float()

    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / S  # Expected probability

    # 1. Error Term (Risk)
    err = torch.sum((y_one_hot - p) ** 2, dim=1, keepdim=True)

    # 2. Variance Term
    var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

    # 3. KL Divergence (Regularization)
    kl = kl_divergence(alpha, num_classes, device)

    # Annealing: Gradually introduce KL term
    annealing_coef = min(1, epoch_num / annealing_step)

    loss = torch.mean(err + var + (annealing_coef * lamb * kl))
    return loss