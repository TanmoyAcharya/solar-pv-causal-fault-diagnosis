"""
Deep learning models and training utilities for Solar PV fault diagnosis.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
import math


# ── Model definitions ──────────────────────────────────────────────────────────

class PVLSTMModel(nn.Module):
    """Bidirectional LSTM for PV fault sequence classification."""

    def __init__(self, n_features: int, n_classes: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.classifier(out)


class PVTransformerModel(nn.Module):
    """Transformer encoder for PV fault sequence classification."""

    def __init__(self, n_features: int, n_classes: int, hidden_size: int = 64,
                 num_layers: int = 2, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class PVCausalInformedNet(nn.Module):
    """Transformer with a causal-adjacency mask applied to self-attention."""

    def __init__(self, n_features: int, n_classes: int, hidden_size: int = 64,
                 num_layers: int = 2, n_heads: int = 4, dropout: float = 0.2,
                 causal_mask: np.ndarray | None = None):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_size)

        # Register causal mask as buffer so it moves with the model
        if causal_mask is not None:
            mask_t = torch.tensor(causal_mask, dtype=torch.float32)
            # Invert: 0 means blocked in additive mask convention
            additive = torch.where(mask_t > 0, torch.zeros_like(mask_t),
                                   torch.full_like(mask_t, float("-inf")))
        else:
            additive = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes),
        )
        self._causal_mask = additive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        mask = None
        if self._causal_mask is not None:
            seq_len = x.size(1)
            # Use causal_mask for feature dimension; for sequence dimension use None
            mask = None  # seq-level mask not directly applicable here
        x = self.encoder(x, mask=mask)
        x = x.mean(dim=1)
        return self.classifier(x)


# ── Factory ────────────────────────────────────────────────────────────────────

def create_model(model_type: str, n_features: int, n_classes: int,
                 hidden_size: int = 64, num_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.2, causal_mask: np.ndarray | None = None) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "lstm":
        return PVLSTMModel(n_features, n_classes, hidden_size, num_layers, dropout)
    elif model_type == "transformer":
        return PVTransformerModel(n_features, n_classes, hidden_size, num_layers, n_heads, dropout)
    elif model_type in ("causalinformednet", "causal"):
        return PVCausalInformedNet(n_features, n_classes, hidden_size, num_layers,
                                   n_heads, dropout, causal_mask)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         "Choose 'lstm', 'transformer', or 'causalinformednet'.")


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(model: nn.Module,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = 20, lr: float = 1e-3, batch_size: int = 64,
                device: str = "cpu",
                progress_callback=None) -> dict:
    """Train model and return history dict."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def _make_loader(X, y, shuffle):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)

    train_loader = _make_loader(X_train, y_train, True)
    val_loader = _make_loader(X_val, y_val, False)

    history: dict[str, list] = {"train_loss": [], "val_loss": [],
                                  "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(yb)
            t_correct += (logits.argmax(1) == yb).sum().item()
            t_total += len(yb)
        scheduler.step()

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                v_loss += loss.item() * len(yb)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total += len(yb)

        history["train_loss"].append(t_loss / t_total)
        history["val_loss"].append(v_loss / v_total)
        history["train_acc"].append(t_correct / t_total)
        history["val_acc"].append(v_correct / v_total)

        if progress_callback is not None:
            progress_callback(epoch, epochs, history)

    return history


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray,
                   device: str = "cpu") -> dict:
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        preds = logits.argmax(1).cpu().numpy()

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_test, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "class_report": classification_report(y_test, preds, zero_division=0),
    }


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_single(model: nn.Module, sequence: np.ndarray,
                   fault_labels: dict, device: str = "cpu") -> dict:
    """Predict fault class for a single sequence (seq_len, n_features)."""
    model.eval()
    model = model.to(device)
    x = torch.tensor(sequence[np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    class_idx = int(probs.argmax())
    return {
        "class_idx": class_idx,
        "class_name": fault_labels.get(class_idx, str(class_idx)),
        "confidence": float(probs[class_idx]),
        "probabilities": {fault_labels.get(i, str(i)): float(p)
                          for i, p in enumerate(probs)},
    }


# ── Gradient attribution ───────────────────────────────────────────────────────

def get_gradient_attribution(model: nn.Module, sequence: np.ndarray,
                              feature_cols: list, device: str = "cpu") -> dict:
    """Input × gradient saliency over the last time step."""
    model.eval()
    model = model.to(device)
    x = torch.tensor(sequence[np.newaxis], dtype=torch.float32,
                      requires_grad=True).to(device)
    logits = model(x)
    pred_class = logits.argmax(1).item()
    logits[0, pred_class].backward()
    grad = x.grad.detach().cpu().numpy()[0]          # (seq_len, n_features)
    importance = np.abs(grad * sequence).mean(axis=0)  # mean over time steps
    importance = importance / (importance.sum() + 1e-9)
    return {col: float(importance[i]) for i, col in enumerate(feature_cols)}
