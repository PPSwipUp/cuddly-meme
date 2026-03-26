"""
Stage 4 — Model Architecture
Trading Algorithm Blueprint

TCN encoder + prediction head, exactly as specified in blueprint Sections 4.1–4.4.

Architecture:
  Input  : [batch, lookback=60, n_features=29]
  Encoder: 3-layer TCN with dilated causal convolutions (dilations 1,2,4)
           Hidden dim 128, kernel 3, GELU, LayerNorm, Dropout 0.25
           Global average pooling → [batch, 128]
  Head   : MLP 128→64→32→output, three parallel heads for horizons 1,5,20
           Directional output : sigmoid (BCE loss)
           Magnitude output   : linear  (Huber loss)

Regularisation:
  Dropout 0.25 (encoder), 0.30 (head)
  L2 weight decay 1e-4 (applied via AdamW)
  Gradient clipping 1.0
  Label smoothing 0.05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── TCN building block ───────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """
    Causal convolution: pads left only so the output at time t depends
    only on inputs at times ≤ t. Required for valid time-series modelling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation   # left-pad only
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,            # we handle padding manually
        )

    def forward(self, x):
        # x: [batch, channels, seq_len]
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    Single TCN residual block:
      CausalConv → LayerNorm → GELU → Dropout →
      CausalConv → LayerNorm → GELU → Dropout →
      residual connection (1×1 conv if channel dims differ)
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels,  out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        # LayerNorm over the channel dimension
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Residual projection if dimensions differ
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        # x: [batch, channels, seq_len]
        residual = self.residual_proj(x)

        out = self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)  # LayerNorm on channels
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)

        return self.activation(out + residual)


# ─── TCN Encoder ─────────────────────────────────────────────────────────────

class TCNEncoder(nn.Module):
    """
    3-layer TCN encoder (blueprint Section 4.2).
    Dilations: 1, 2, 4 → receptive field covers 7 bars per layer.
    Input : [batch, lookback, n_features]
    Output: [batch, hidden_dim]  (global average pooled)
    """
    def __init__(
        self,
        n_features:  int = 29,
        hidden_dim:  int = 128,
        n_layers:    int = 3,
        kernel_size: int = 3,
        dropout:     float = 0.25,
    ):
        super().__init__()
        assert hidden_dim <= 256, "Blueprint cap: hidden_dim must not exceed 256"

        dilations  = [2 ** i for i in range(n_layers)]   # [1, 2, 4]
        layers     = []
        in_ch      = n_features

        for dil in dilations:
            layers.append(TCNBlock(in_ch, hidden_dim, kernel_size, dil, dropout))
            in_ch = hidden_dim

        self.network    = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: [batch, seq_len, n_features]  →  transpose for Conv1d
        x = x.transpose(1, 2)             # [batch, n_features, seq_len]
        x = self.network(x)               # [batch, hidden_dim, seq_len]
        x = x.mean(dim=2)                 # global average pool → [batch, hidden_dim]
        return x


# ─── Prediction Head ─────────────────────────────────────────────────────────

class PredictionHead(nn.Module):
    """
    MLP head for a single horizon (blueprint Section 4.3).
    Produces:
      direction : sigmoid probability that Close[t+h] > Close[t]
      magnitude : predicted log return over horizon h (linear, no activation)
    """
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.30):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)

        self.dir_out = nn.Linear(32, 1)   # directional probability
        self.mag_out = nn.Linear(32, 1)   # magnitude (log return)

        self.dropout    = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.activation(self.fc2(x))

        direction = torch.sigmoid(self.dir_out(x)).squeeze(-1)   # [batch]
        magnitude = self.mag_out(x).squeeze(-1)                   # [batch]
        return direction, magnitude


# ─── Full model ───────────────────────────────────────────────────────────────

class TradingModel(nn.Module):
    """
    Complete model: TCNEncoder + three parallel PredictionHeads for
    horizons [1, 5, 20] bars (blueprint Sections 4.2–4.3).

    Forward returns a dict:
      {
        1:  (direction_prob, magnitude),
        5:  (direction_prob, magnitude),
        20: (direction_prob, magnitude),
      }
    """
    HORIZONS = [1, 5, 20]

    def __init__(
        self,
        n_features:  int   = 29,
        hidden_dim:  int   = 128,
        n_layers:    int   = 3,
        kernel_size: int   = 3,
        enc_dropout: float = 0.25,
        head_dropout: float = 0.30,
    ):
        super().__init__()

        self.encoder = TCNEncoder(
            n_features=n_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=enc_dropout,
        )

        self.heads = nn.ModuleDict({
            str(h): PredictionHead(hidden_dim=hidden_dim, dropout=head_dropout)
            for h in self.HORIZONS
        })

    def forward(self, x):
        """x: [batch, seq_len, n_features]"""
        encoded = self.encoder(x)
        return {
            h: self.heads[str(h)](encoded)
            for h in self.HORIZONS
        }

    def encoder_params(self):
        return self.encoder.parameters()

    def head_params(self):
        return self.heads.parameters()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Loss functions ───────────────────────────────────────────────────────────

class TradingLoss(nn.Module):
    """
    Combined loss for one horizon (blueprint Section 4.3 & 5.3):
      0.5 × BCE(direction, label_smoothing=0.05)
      0.5 × Huber(magnitude, delta=0.01)
    """
    def __init__(self, label_smoothing: float = 0.05, huber_delta: float = 0.01):
        super().__init__()
        self.smoothing   = label_smoothing
        self.huber_delta = huber_delta

    def forward(self, direction_pred, magnitude_pred, direction_true, magnitude_true):
        # Label smoothing: pull targets away from 0 and 1
        smooth_true = direction_true * (1 - self.smoothing) + 0.5 * self.smoothing
        bce  = F.binary_cross_entropy(direction_pred, smooth_true)
        huber = F.huber_loss(magnitude_pred, magnitude_true, delta=self.huber_delta)
        return 0.5 * bce + 0.5 * huber, bce, huber


class MultiHorizonLoss(nn.Module):
    """Averages TradingLoss equally across all three horizons."""
    def __init__(self):
        super().__init__()
        self.horizon_loss = TradingLoss()

    def forward(self, predictions, targets):
        """
        predictions: dict {horizon: (direction_pred, magnitude_pred)}
        targets:     dict {horizon: (direction_true, magnitude_true)}
        Returns: (total_loss, bce_loss, huber_loss)
        """
        total = bce_total = huber_total = 0.0
        n = len(predictions)

        for h, (dir_pred, mag_pred) in predictions.items():
            dir_true, mag_true = targets[h]
            loss, bce, huber   = self.horizon_loss(dir_pred, mag_pred, dir_true, mag_true)
            total      += loss
            bce_total  += bce
            huber_total += huber

        return total / n, bce_total / n, huber_total / n


# ─── Model factory ────────────────────────────────────────────────────────────

def build_model(n_features: int = 29, device: str = "cpu") -> TradingModel:
    """Instantiate model with blueprint-specified default hyperparameters."""
    model = TradingModel(
        n_features=n_features,
        hidden_dim=128,
        n_layers=3,
        kernel_size=3,
        enc_dropout=0.25,
        head_dropout=0.30,
    )
    return model.to(device)


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = build_model(n_features=29, device=device)
    print(f"Parameters: {model.count_parameters():,}")

    # Verify forward pass with blueprint dimensions
    batch   = torch.randn(256, 60, 29).to(device)   # [batch=256, lookback=60, features=29]
    outputs = model(batch)

    print("\nForward pass output shapes:")
    for h, (direction, magnitude) in outputs.items():
        print(f"  Horizon {h:>2}:  direction {tuple(direction.shape)}  "
              f"magnitude {tuple(magnitude.shape)}")

    # Verify loss computation
    criterion = MultiHorizonLoss()
    fake_targets = {
        h: (torch.rand(256).to(device), torch.randn(256).to(device) * 0.01)
        for h in TradingModel.HORIZONS
    }
    total_loss, bce, huber = criterion(outputs, fake_targets)
    print(f"\nLoss check:  total={total_loss.item():.4f}  "
          f"BCE={bce.item():.4f}  Huber={huber.item():.4f}")

    # Verify gradient flows through encoder and all heads
    total_loss.backward()
    grads_ok = all(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"Gradients flow to all parameters: {grads_ok}")

    print("\nStage 4 model architecture — OK")