from typing import Callable, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Helper utilities for the sequence model
# =============================================================================

DEFAULT_SEQ_LEN = 15  # **must be odd** so the target is centred
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------#
# 1. Sequence helpers (centre-aligned vs. causal)
# -----------------------------------------------------------------------------#
def _create_sequences_center(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Return (seq, target) where target is the *centre* point of each window."""
    half      = seq_len // 2
    idx_start = half
    idx_end   = len(X) - half
    seqs, tgs = [], []
    for i in range(idx_start, idx_end):
        seqs.append(X[i - half : i + half + 1].T)     # (C, T)
        tgs.append(y[i])
    return np.asarray(seqs, dtype=np.float32), np.asarray(tgs, dtype=np.float32)


def _create_sequences_tail(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Return (seq, target) where target is the *last* point (causal window)."""
    offset    = seq_len - 1
    seqs, tgs = [], []
    for i in range(offset, len(X)):
        seqs.append(X[i - offset : i + 1].T)          # (C, T)
        tgs.append(y[i])
    return np.asarray(seqs, dtype=np.float32), np.asarray(tgs, dtype=np.float32)


# -----------------------------------------------------------------------------#
# 2. A lightweight Dataset wrapper
# -----------------------------------------------------------------------------#
class TimeSeriesDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        self.X = torch.from_numpy(X_seq)              # (N, C, T)
        self.y = torch.from_numpy(y_seq)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------------------------------------------------------#
# 3. A single TCN block – residual, dilation & causal all optional
# -----------------------------------------------------------------------------#
class _TCNBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        *,
        causal: bool,
        residual: bool,
    ):
        super().__init__()
        self.causal   = causal
        self.residual = residual
        self.left_pad = (kernel_size - 1) * dilation if causal else 0
        sym_pad       = dilation * ((kernel_size - 1) // 2)

        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            dilation=dilation,
            padding=0 if causal else sym_pad
        )
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size,
            dilation=dilation,
            padding=0 if causal else sym_pad
        )

        self.relu1, self.relu2 = nn.ReLU(), nn.ReLU()
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_ch)

        self.downsample = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if residual and (in_ch != out_ch) else None
        )

    # --------------------------------------------------------------------- #
    def _maybe_left_pad(self, x):
        return F.pad(x, (self.left_pad, 0)) if self.causal and self.left_pad else x

    # --------------------------------------------------------------------- #
    def forward(self, x):
        out = self.conv1(self._maybe_left_pad(x))
        out = self.drop1(self.relu1(out))

        out = self.conv2(self._maybe_left_pad(out))

        if self.residual:
            res = x if self.downsample is None else self.downsample(x)
            out = out + res

        out = self.drop2(self.relu2(out))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out


# -----------------------------------------------------------------------------#
# 4. The full (flexible) TCN
# -----------------------------------------------------------------------------#
class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_ch: int = 48,
        kernel_size: int = 5,
        num_layers: int = 2,
        dropout: float = 0.15,
        out_dim: int = 1,
        *,
        causal: bool = False,
        use_residual: bool = True,
        use_dilation: bool = True,
    ):
        super().__init__()
        layers, in_ch = [], num_features
        for i in range(num_layers):
            dilation = (2 ** i) if use_dilation else 1
            layers.append(
                _TCNBlock(
                    in_ch, hidden_ch, kernel_size, dilation, dropout,
                    causal=causal, residual=use_residual
                )
            )
            in_ch = hidden_ch

        self.tcn   = nn.Sequential(*layers)
        self.fc    = nn.Linear(hidden_ch, out_dim)
        self.causal = causal
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    # --------------------------------------------------------------------- #
    def forward(self, x):                           # x: (B, C, T)
        y = self.tcn(x)                             # (B, H, T)
        idx = -1 if self.causal else x.size(2) // 2
        return self.fc(y[:, :, idx])                # (B, out_dim)


# -----------------------------------------------------------------------------#
# 5. sklearn-style wrapper (keeps .fit / .predict signature)
# -----------------------------------------------------------------------------#
class TCNRegressorWrapper:
    """
    A thin wrapper exposing .fit / .predict with NumPy I/O so the model
    can slot into any registry expecting a 'scikit-like' interface.
    """

    def __init__(
        self,
        # ---- data / training -------------------------------------------------
        seq_len: int = DEFAULT_SEQ_LEN,
        batch_size: int = 128,
        epochs: int = 30,
        learning_rate: float = 1e-4,
        optimizer_type: str = "rmsprop",
        # ---- architecture ----------------------------------------------------
        hidden_channels: int = 48,
        kernel_size: int = 5,
        num_layers: int = 2,
        dropout: float = 0.15,
        causal: bool = False,
        use_residual: bool = True,
        use_dilation: bool = True,
        # ---- misc ------------------------------------------------------------
        undersample_ratio: int | None = None,
        seed: int = 42,
    ):
        self.seq_len          = seq_len
        self.batch_size       = batch_size
        self.epochs           = epochs
        self.learning_rate    = learning_rate
        self.optimizer_type   = optimizer_type.lower()

        self.hidden_channels  = hidden_channels
        self.kernel_size      = kernel_size
        self.num_layers       = num_layers
        self.dropout          = dropout

        self.causal           = causal
        self.use_residual     = use_residual
        self.use_dilation     = use_dilation

        self.undersample_ratio = undersample_ratio
        self.seed              = seed

        self.device = DEVICE
        self.model: TemporalConvNet | None = None

    # --------------------------------------------------------------------- #
    def _make_sequences(self, X: np.ndarray, y: np.ndarray):
        if self.causal:
            return _create_sequences_tail(X, y, self.seq_len)
        return _create_sequences_center(X, y, self.seq_len)

    # --------------------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the TCN on *time-ordered* data (X, y).

        X shape: (N, C)  – C = num features
        y shape: (N,)    – scalar targets
        """
        # -- optional undersampling of zero targets --------------------------
        if isinstance(self.undersample_ratio, int) and self.undersample_ratio > 1:
            np.random.seed(self.seed)
            idx_zero    = np.where(y == 0)[0]
            idx_nonzero = np.where(y != 0)[0]
            if len(idx_zero):
                keep_z   = np.random.choice(
                    idx_zero, max(1, len(idx_zero) // self.undersample_ratio), replace=False
                )
                keep_idx = np.concatenate([keep_z, idx_nonzero])
                np.random.shuffle(keep_idx)
                X, y = X[keep_idx], y[keep_idx]
                print(f"TCN_U: undersampled zeros {len(idx_zero)} → {len(keep_z)}")

        # -- window the data --------------------------------------------------
        X_seq, y_seq = self._make_sequences(X, y)
        dataset      = TimeSeriesDataset(X_seq, y_seq)
        loader       = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # -- build the network -----------------------------------------------
        num_features = X.shape[1]
        self.model = TemporalConvNet(
            num_features       = num_features,
            hidden_ch          = self.hidden_channels,
            kernel_size        = self.kernel_size,
            num_layers         = self.num_layers,
            dropout            = self.dropout,
            out_dim            = 1,
            causal             = self.causal,
            use_residual       = self.use_residual,
            use_dilation       = self.use_dilation,
        ).to(self.device)

        criterion = nn.MSELoss()
        optim_cls = torch.optim.Adam if self.optimizer_type == "adam" else torch.optim.RMSprop
        optimizer = optim_cls(self.model.parameters(), lr=self.learning_rate)

        # -- training loop ----------------------------------------------------
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(-1)   # (B, 1)

                optimizer.zero_grad()
                loss = criterion(self.model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # print(f"Epoch {epoch+1}/{self.epochs}  loss={epoch_loss/len(loader):.4f}")

        return self

    # --------------------------------------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict for every position of the input sequence X.

        Output length == input length (front / back edges are filled with
        the first / last available prediction for convenience).
        """
        if self.model is None:
            raise RuntimeError("Model not trained – call .fit() first.")

        offset = self.seq_len - 1 if self.causal else self.seq_len // 2
        n      = len(X)

        preds  = np.zeros(n, dtype=np.float32)
        filled = np.zeros(n, dtype=bool)

        X_seq, _ = self._make_sequences(X, np.zeros(n, dtype=np.float32))
        loader   = DataLoader(torch.from_numpy(X_seq).float(),
                              batch_size=self.batch_size, shuffle=False)

        idx = 0
        self.model.eval()
        with torch.no_grad():
            for batch_X in loader:
                out = self.model(batch_X.to(self.device))
                out = out.cpu().numpy().squeeze(-1)
                for val in out:
                    pos = idx + offset
                    preds[pos]  = val
                    filled[pos] = True
                    idx += 1

        # -- edge filling so len(preds) == len(X) ----------------------------
        if np.any(filled):
            first_val = preds[filled][0]
            last_val  = preds[filled][-1]
            preds[:offset]  = first_val
            preds[idx+offset:] = last_val

        return preds


# Factory helpers -------------------------------------------------------------

def tcn_regressor_default() -> TCNRegressorWrapper:
    return TCNRegressorWrapper()

def tcn_hidden24() -> TCNRegressorWrapper:
    return TCNRegressorWrapper(hidden_channels=24)

def tcn_hidden60() -> TCNRegressorWrapper:
    return TCNRegressorWrapper(hidden_channels=60)

def tcn_seq25() -> TCNRegressorWrapper:
    return TCNRegressorWrapper(seq_len=25, hidden_channels=24)

def tcn_seq29_l3_k3() -> TCNRegressorWrapper:
    return TCNRegressorWrapper(seq_len=29, num_layers=3, kernel_size=3, hidden_channels=24)

def tcn_seq5_l1_k3() -> TCNRegressorWrapper:
    return TCNRegressorWrapper(seq_len=5, num_layers=1, kernel_size=3, hidden_channels=24)

def tcn_adam() -> TCNRegressorWrapper:
    return TCNRegressorWrapper(optimizer_type="adam", hidden_channels=24)

def tcn_no_dilation() -> TCNRegressorWrapper:
    return TCNRegressorWrapper(use_dilation=False, hidden_channels=24)

def tcn_no_residual() -> TCNRegressorWrapper:
    return TCNRegressorWrapper(use_residual=False, hidden_channels=24)

def tcn_causal() -> TCNRegressorWrapper:
    return TCNRegressorWrapper(causal=True, hidden_channels=24)


# -----------------------------------------------------------------------------
# Registry — evaluate.py imports this directly
# -----------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Callable[[], object]] = {
    "TCN": tcn_regressor_default,
    "TCN_hidden24": tcn_hidden24,
    "TCN_hidden60": tcn_hidden60,
    "TCN_seq25": tcn_seq25,
    "TCN_seq29_l3_k3": tcn_seq29_l3_k3,
    "TCN_seq5_l1_k3": tcn_seq5_l1_k3,
    "TCN_adam": tcn_adam,
    "TCN_no_dilation": tcn_no_dilation,
    "TCN_no_residual": tcn_no_residual,
    "TCN_causal": tcn_causal,
}