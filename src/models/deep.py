import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # take last time-step
        out = out[:, -1, :]
        return self.head(out).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)
        # Transformer expects (seq_len, batch, d_model)
        x = x.permute(1, 0, 2)
        out = self.encoder(x)
        out = out[-1]  # last time-step
        return self.head(out).squeeze(-1)


def train_torch_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, epochs: int = 20, batch_size: int = 64, lr: float = 1e-3, device: str = None, ckpt_path: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                xb = torch.tensor(X_val, dtype=torch.float32).to(device)
                yb = torch.tensor(y_val, dtype=torch.float32).to(device)
                preds = model(xb)
                val_loss = loss_fn(preds, yb).item()
            history['val_loss'].append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                if ckpt_path:
                    torch.save(model.state_dict(), ckpt_path)

    return model, history
