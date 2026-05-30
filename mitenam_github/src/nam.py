"""
Neural Additive Model (NAM) + Knowledge Distillation Loss
==========================================================

NAM architecture (Agarwal et al., NeurIPS 2021):
    Per-feature independent subnetwork:
        1 -> 64 -> 64 -> 32 -> 1
    Output logit = bias + sum of per-feature contributions

KD loss (Hinton et al., 2015):
    L = α * BCE(z_s, y) + (1 - α) * T² * KD(σ(z_s/T), σ(z_t/T))

In the binary classification setting, the soft-target cross-entropy
between sigmoid-of-temperature-scaled logits is equivalent (up to a
constant) to the KL divergence with respect to the student parameters.
"""

import numpy as np
import torch
import torch.nn as nn


# Hyperparameters (paper Section 2.5)
HIDDEN_DIMS = (64, 64, 32)  # NAM per-feature subnetwork
DROPOUT = 0.1
ALPHA = 0.5                  # BCE weight in joint loss
TEMPERATURE = 2.0            # KD temperature T
WEIGHT_DECAY = 1e-5          # L2 regularization on optimizer
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
MAX_EPOCHS = 80
PATIENCE = 12                # Early stopping on validation AUROC


class NAM(nn.Module):
    """
    Neural Additive Model: sum of per-feature independent subnetworks.

    Each feature is processed by an independent 3-layer MLP
    (1 -> 64 -> 64 -> 32 -> 1) with ReLU and dropout.

    Output logit z_s(x) = b + sum_j f_j(x_j)
    """

    def __init__(self, num_features: int,
                 hidden: tuple = HIDDEN_DIMS,
                 dropout: float = DROPOUT):
        super().__init__()
        feature_nns = []
        for _ in range(num_features):
            feature_nns.append(nn.Sequential(
                nn.Linear(1, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden[1], hidden[2]), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden[2], 1),
            ))
        self.feature_nns = nn.ModuleList(feature_nns)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch,).
        """
        out = self.bias.expand(x.size(0), 1)
        for i, fnn in enumerate(self.feature_nns):
            out = out + fnn(x[:, i:i + 1])
        return out.squeeze(-1)


def kd_loss_binary(z_s: torch.Tensor,
                   p_teacher: torch.Tensor,
                   T: float = TEMPERATURE) -> torch.Tensor:
    """
    Knowledge distillation loss for binary classification.

    KD(σ(z_s/T), σ(z_t/T)) computed as KL divergence between
    temperature-scaled sigmoid outputs of student and teacher.

    The output is multiplied by T² (standard Hinton scaling).

    Parameters
    ----------
    z_s : torch.Tensor
        Student logits, shape (batch,).
    p_teacher : torch.Tensor
        Teacher ensemble probabilities (in [0, 1]), shape (batch,).
    T : float
        Temperature.

    Returns
    -------
    torch.Tensor
        Scalar KD loss (already scaled by T²).
    """
    eps = 1e-7
    p_t = p_teacher.clamp(eps, 1 - eps)
    z_t = torch.log(p_t / (1 - p_t))     # teacher logit
    q_s = torch.sigmoid(z_s / T).clamp(eps, 1 - eps)
    q_t = torch.sigmoid(z_t / T).clamp(eps, 1 - eps)
    kl = q_s * (torch.log(q_s) - torch.log(q_t)) + \
         (1 - q_s) * (torch.log(1 - q_s) - torch.log(1 - q_t))
    return kl.mean() * (T ** 2)
