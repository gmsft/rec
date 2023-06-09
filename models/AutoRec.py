import numpy as np
import torch
from torch import nn


class AutoRec(nn.Module):
    def __init__(self, num_user, latent_dim, dropout):
        super().__init__()
        self.encoder = nn.Linear(num_user, latent_dim)
        self.decoder = nn.Linear(latent_dim, num_user)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.encoder(x)
        # out = self.relu(out)
        out = self.sig(out)
        out = self.dropout(out)
        out = self.decoder(out)
        # out = self.relu(out)
        # Mask the gradient of unobserved user-item interaction during training
        if torch.is_grad_enabled():
            out = out * torch.sign(x)
        return out

