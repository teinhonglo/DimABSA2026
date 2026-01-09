import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from transformers import AutoModel, AutoConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F

import os
from transformers.models.wav2vec2 import Wav2Vec2PreTrainedModel

class MSAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_levels=(32, 64, 128)):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.sparsity_levels = sparsity_levels

    def forward(self, x):
        # x: [B, D]
        x = self.fc(x)
        sparse_outputs = []
        for k in self.sparsity_levels:
            topk = torch.topk(x, k=k, dim=-1)
            mask = torch.zeros_like(x).scatter(-1, topk.indices, 1.0)
            sparse_out = F.relu(x * mask)
            sparse_outputs.append(sparse_out)
        return sparse_outputs  # List of [B, D]

class MSAEDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        return self.fc(z)  # [B, D]

class MSAEWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_levels=(32, 64, 128), alpha=None):
        super().__init__()
        self.encoder = MSAEEncoder(input_dim, hidden_dim, sparsity_levels)
        self.decoder = MSAEDecoder(hidden_dim, input_dim)
        self.sparsity_levels = sparsity_levels
        self.alpha = alpha if alpha is not None else [1.0 for _ in sparsity_levels]

    def forward(self, x):
        sparse_list = self.encoder(x)  # list of [B, D]
        loss = 0.0
        recons = []
        for i, z in enumerate(sparse_list):
            x_hat = self.decoder(z)
            recons.append(x_hat)
            loss += self.alpha[i] * F.mse_loss(x_hat, x)
        return sparse_list[-1], loss  # 最後一層當作 sparse representation