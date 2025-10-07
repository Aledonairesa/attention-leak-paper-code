from typing import Callable, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


# =============================================================================
# Helper utilities for the sequence model
# =============================================================================

DEFAULT_SEQ_LEN = 15  # **must be odd** so the target is centred


def create_sequences_center(X: np.ndarray, y: np.ndarray, seq_len: int = DEFAULT_SEQ_LEN) -> Tuple[np.ndarray, np.ndarray]:
    """Create overlapping centred windows of length *seq_len*.

    Returns
    -------
    X_seq : ndarray, shape (n_seq, n_features, seq_len)
    y_seq : ndarray, shape (n_seq,)
    """
    X_seq, y_seq = [], []
    half = seq_len // 2
    for i in range(half, len(X) - half):
        X_seq.append(X[i - half : i + half + 1].T)  # channels‑first
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


class TimeSeriesDataset(Dataset):
    """PyTorch dataset wrapping the (sequence, target) pairs."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FocalLoss(nn.Module):
    """Focal Loss for binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the probability of the true class.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()  # Ensure targets are float for BCEWithLogitsLoss
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs) # p_t
        
        focal_weight = (1 - pt).pow(self.gamma)
        
        # Apply alpha weight: alpha for positive class, (1-alpha) for negative class
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_t * focal_weight * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# =============================================================================
# Temporal Convolutional Network (non‑causal, symmetric padding)
# =============================================================================

class ResidualBlock_NonCausal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        if self.downsample is not None:
            res = self.downsample(res)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = out + res
        out = self.relu2(out)
        out = self.dropout2(out)
        out = out.transpose(1, 2)  # (B, T, C)
        out = self.layer_norm(out)
        out = out.transpose(1, 2)  # back to (B, C, T)
        return out


class TCN_NonCausal(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 48,
        kernel_size: int = 5,
        num_layers: int = 2,
        dropout: float = 0.15,
        out_dim: int = 1,
    ):
        super().__init__()
        layers = []
        in_channels = num_features
        for i in range(num_layers):
            dilation = 2**i
            padding = dilation * ((kernel_size - 1) // 2)
            layers.append(
                ResidualBlock_NonCausal(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )
            in_channels = hidden_channels
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_channels, out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x: (B, C, T)
        out = self.tcn(x)
        center = x.size(2) // 2
        out = out[:, :, center]
        return self.fc(out)


# =============================================================================
# Wrapper so that the TCN can plug into evaluate.py like a sklearn regressor
# =============================================================================

class TCNRegressorWrapper:
    """Self-contained wrapper exposing .fit / .predict with numpy inputs."""

    def __init__(
        self,
        seq_len: int = DEFAULT_SEQ_LEN,
        batch_size: int = 128,
        hidden_channels: int = 48,
        kernel_size: int = 5,
        num_layers: int = 2,
        dropout: float = 0.15,
        epochs: int = 30,
        learning_rate: float = 1e-4,
        optimizer_type: str = "rmsprop",
        undersample_ratio: Optional[int] = None,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: TCN_NonCausal | None = None

        # New parameters for undersampling
        self.undersample_ratio = undersample_ratio
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the TCN on the given (time-ordered) data."""

        # — optionally undersample all-zero targets before sequence creation —
        if isinstance(self.undersample_ratio, int) and self.undersample_ratio > 1:
            np.random.seed(self.seed)
            idx_zero     = np.where(y == 0)[0]
            idx_nonzero  = np.where(y != 0)[0]
            if len(idx_zero) > 0:
                n_keep_zero = max(1, len(idx_zero) // self.undersample_ratio)
                chosen_zero = np.random.choice(idx_zero, n_keep_zero, replace=False)
                keep_idx = np.concatenate([chosen_zero, idx_nonzero])
                np.random.shuffle(keep_idx)
                X, y = X[keep_idx], y[keep_idx]
                print(f"TCN_U: undersampled zeros {len(idx_zero)} → {n_keep_zero}")

        # — rest of your existing training code unchanged —
        X_seq, y_seq = create_sequences_center(X, y, self.seq_len)
        dataset = TimeSeriesDataset(X_seq, y_seq)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        num_features = X.shape[1]
        self.model = TCN_NonCausal(
            num_features=num_features,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.MSELoss()
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(-1)
                optimizer.zero_grad()
                out = self.model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(loader):.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not trained yet. Call .fit first."
        self.model.eval()

        half = self.seq_len // 2
        n = len(X)
        preds = np.zeros(n, dtype=np.float32)
        filled = np.zeros(n, dtype=bool)

        X_seq, _ = create_sequences_center(X, np.zeros(n), self.seq_len)
        loader = DataLoader(torch.from_numpy(X_seq).float(), batch_size=self.batch_size, shuffle=False)

        idx = 0
        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X.to(self.device)
                out = self.model(batch_X).cpu().numpy().squeeze(-1)
                for val in out:
                    pos = idx + half
                    preds[pos] = val
                    filled[pos] = True
                    idx += 1

        # Fill the edges so output length == input length
        if np.any(filled):
            first_val = preds[filled].flat[0]
            last_val  = preds[filled].flat[-1]
            preds[:half] = first_val
            preds[-half:] = last_val

        return preds


# =============================================================================
# Binary Classification wrapper around the TCN
# =============================================================================

class TCNClassifierWrapper:
    """Wrapper providing .fit / .predict / .predict_proba for binary classification (2 classes)."""

    def __init__(
        self,
        seq_len: int = DEFAULT_SEQ_LEN,
        batch_size: int = 128,
        hidden_channels: int = 48,
        kernel_size: int = 5,
        num_layers: int = 2,
        dropout: float = 0.15,
        epochs: int = 30,
        learning_rate: float = 1e-4,
        optimizer_type: str = "rmsprop",
        threshold: float = 0.5,  # decision threshold on probability
        loss_type: str = "bce",  # Supports "bce", "focal", "weighted_bce", "undersampled_weighted_bce", "undersampled_bce"
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type.lower()
        self.threshold = threshold
        self.loss_type = loss_type.lower()
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: TCN_NonCausal | None = None

    @staticmethod
    def _clip_y(y: np.ndarray) -> np.ndarray:
        """Clip integer labels to {0,1}, mapping 2,3,... -> 1."""
        return np.clip(y, 0, 1).astype(np.float32)

    # ------------------------------------------------------------------
    # scikit‑like API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the TCN classifier on ordered data for two‑class problems."""
        y_clipped = self._clip_y(y)

        X_for_processing = X
        y_for_processing = y_clipped

        # --- START Undersampling Data Transformation (if applicable) ---
        if self.loss_type == "undersampled_weighted_bce" or self.loss_type == "undersampled_bce": # <--- MODIFIED CONDITION
            np.random.seed(42) 
            
            indices_class0 = np.where(y_for_processing == 0)[0]
            indices_class1 = np.where(y_for_processing == 1)[0]
            n_class0_original = len(indices_class0)
            
            if n_class0_original > 0:
                n_class0_target = max(1, int(n_class0_original / 5))
                
                if n_class0_target < n_class0_original:
                    chosen_class0_indices = np.random.choice(indices_class0, size=n_class0_target, replace=False)
                else: 
                    chosen_class0_indices = indices_class0
                
                final_indices = np.concatenate([chosen_class0_indices, indices_class1])
                np.random.shuffle(final_indices)

                X_for_processing = X[final_indices]
                y_for_processing = y_clipped[final_indices]
                
                print(f"{self.loss_type.upper()}: Undersampled class 0 from {n_class0_original} to {len(chosen_class0_indices)}. Class 1 count: {len(indices_class1)}.")
            else:
                print(f"{self.loss_type.upper()}: No class 0 samples to undersample.")
        # --- END Undersampling Data Transformation ---

        if len(y_for_processing) < self.seq_len:
             print(f"Warning: After processing for {self.loss_type.upper()}, data length ({len(y_for_processing)}) is less than sequence length ({self.seq_len}). No sequences will be created, model will not train.")
        
        X_seq, y_seq = create_sequences_center(X_for_processing, y_for_processing, self.seq_len)
        dataset = TimeSeriesDataset(X_seq, y_seq)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        num_features = X_for_processing.shape[1] if X_for_processing.ndim > 1 and X_for_processing.shape[0] > 0 else (X.shape[1] if X.ndim > 1 else 1)
        self.model = TCN_NonCausal(
            num_features=num_features,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            out_dim=1,
        ).to(self.device)

        # Criterion selection
        if self.loss_type == "bce" or self.loss_type == "undersampled_bce": 
            criterion = nn.BCEWithLogitsLoss() # Standard BCE for 'bce' and 'undersampled_bce'
        elif self.loss_type == "focal":
            criterion = FocalLoss(alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma)
        elif self.loss_type == "weighted_bce" or self.loss_type == "undersampled_weighted_bce":
            num_samples_total_processed = len(y_for_processing)
            if num_samples_total_processed == 0:
                pos_weight_value = 1.0
                print(f"Warning: {self.loss_type.upper()} data is empty after processing. Using pos_weight=1.0.")
            else:
                num_positives_processed = np.sum(y_for_processing == 1)
                num_negatives_processed = num_samples_total_processed - num_positives_processed
                
                if num_positives_processed == 0:
                    pos_weight_value = 1.0 
                    print(f"Warning: No positive samples found for {self.loss_type.upper()} after processing. Using pos_weight=1.0.")
                elif num_negatives_processed == 0:
                    pos_weight_value = 1.0
                    print(f"Warning: No negative samples found for {self.loss_type.upper()} after processing. Using pos_weight=1.0.")
                else:
                    pos_weight_value = num_negatives_processed / num_positives_processed
            
            pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            raise ValueError(
                f"Unsupported loss_type: {self.loss_type}. "
                f"Choose 'bce', 'focal', 'weighted_bce', 'undersampled_weighted_bce', or 'undersampled_bce'." # <--- UPDATED MESSAGE
            )
        
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else: # default to RMSprop
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        if not loader:
            print(f"Warning: DataLoader is empty for {self.loss_type.upper()}. Model will not be trained.")
        else:
            for _ in range(self.epochs):
                for batch_X, batch_y in loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device).unsqueeze(-1)
                    optimizer.zero_grad()
                    logits = self.model(batch_X)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
        return self

    def _predict_logits_full(self, X: np.ndarray) -> np.ndarray:
        """Helper running inference on the full sequence, returning logits aligned with X."""
        assert self.model is not None, "Model not trained yet. Call .fit first."
        self.model.eval()
        half = self.seq_len // 2
        n = len(X)
        logits_full = np.zeros(n, dtype=np.float32)
        filled = np.zeros(n, dtype=bool)

        X_seq, _ = create_sequences_center(X, np.zeros(n), self.seq_len)
        loader = DataLoader(torch.from_numpy(X_seq).float(), batch_size=self.batch_size, shuffle=False)
        start_idx = half  # first centre index
        idx = 0
        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X.to(self.device)
                out = self.model(batch_X).cpu().numpy()
                for p in out:
                    pos = start_idx + idx
                    logits_full[pos] = p
                    filled[pos] = True
                    idx += 1
        # Fill edges with nearest logits to maintain length
        if np.any(filled):
            first_val = logits_full[filled].flat[0]
            last_val = logits_full[filled].flat[-1]
            logits_full[:half] = first_val
            logits_full[-half:] = last_val
        return logits_full

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape (n_samples, 2)."""
        logits = self._predict_logits_full(X)
        probs_pos = 1 / (1 + np.exp(-logits))
        probs_neg = 1 - probs_pos
        return np.vstack([probs_neg, probs_pos]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary class predictions (0/1)."""
        probs_pos = self.predict_proba(X)[:, 1]
        return (probs_pos >= self.threshold).astype(int)


# =============================================================================
# Traditional models (Random Forest & KNN)
# =============================================================================


def knn_regressor() -> KNeighborsRegressor:
    """50‑neighbour distance‑weighted KNN regressor."""
    return KNeighborsRegressor(n_neighbors=50, weights="distance")


def random_forest_regressor() -> RandomForestRegressor:
    """Random Forest regressor with 75 trees (depth = 8) for reproducibility."""
    return RandomForestRegressor(
        n_estimators=75,
        max_depth=8,
        #class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


# -----------------------------------------------------------------------------
# 4‑class Random Forest classifier – clip training targets to {0, 1, 2, 3}
# -----------------------------------------------------------------------------

class _ClipMixin:
    """Utility mix-in that clips y ∈ N to the range [0, 3] during *training* only."""

    @staticmethod
    def _clip_y(y):
        return np.clip(y, 0, 3).astype(int)


class RFClassifierWrapper(_ClipMixin):
    """Balanced Random Forest limited to 4 classes (0-3)."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=75,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    # scikit‑learn API ---------------------------------------------------------
    def fit(self, X, y):
        y_c = self._clip_y(y)
        self.model.fit(X, y_c)
        return self

    def predict(self, X):
        return self.model.predict(X)


# -----------------------------------------------------------------------------
# Poisson XGBoost regressor (original)
# -----------------------------------------------------------------------------

class XGBPoissonWrapper:
    """Wrapper for XGBoost Poisson regressor (original behaviour)."""

    def __init__(
        self,
        params: Dict | None = None,
        num_boost_round: int = 300,
        early_stopping_rounds: int = 10,
    ):
        self.params = params or {
            "objective": "count:poisson",
            "eval_metric": "rmse",
            "eta": 0.125,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "lambda": 0.5,
            "alpha": 4,
            "seed": 42,
        }
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, "train")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )
        return self

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)


# Factory helpers -------------------------------------------------------------

def random_forest_classifier() -> RFClassifierWrapper:
    return RFClassifierWrapper()


def xgboost_poisson() -> XGBPoissonWrapper:
    return XGBPoissonWrapper()


def tcn_regressor() -> TCNRegressorWrapper:
    return TCNRegressorWrapper()

def tcn_regressor_u() -> TCNRegressorWrapper:
    """TCN regressor with 1/20 undersampling of y == 0."""
    return TCNRegressorWrapper(undersample_ratio=2)

def tcn_classifier() -> TCNClassifierWrapper:
    return TCNClassifierWrapper()

def tcn_classifier_fl() -> TCNClassifierWrapper:
    """Instantiates TCNClassifierWrapper with Focal Loss (default alpha=0.25, gamma=2.0)."""
    return TCNClassifierWrapper(
        loss_type="focal", 
        focal_loss_alpha=0.25,  # Default Focal Loss alpha
        focal_loss_gamma=2.0    # Default Focal Loss gamma
    )

def tcn_classifier_wl() -> TCNClassifierWrapper:
    """Instantiates TCNClassifierWrapper with dynamically weighted BCE loss."""
    return TCNClassifierWrapper(loss_type="weighted_bce")

def tcn_classifier_u_wl() -> TCNClassifierWrapper:
    """Instantiates TCNClassifierWrapper with undersampling of class 0 (10x reduction) 
    and dynamically weighted BCE loss."""
    return TCNClassifierWrapper(loss_type="undersampled_weighted_bce")

def tcn_classifier_u() -> TCNClassifierWrapper:
    """Instantiates TCNClassifierWrapper with undersampling of class 0 (10x reduction)
    and standard BCE loss."""
    return TCNClassifierWrapper(loss_type="undersampled_bce")


# -----------------------------------------------------------------------------
# Registry — evaluate.py imports this directly
# -----------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Callable[[], object]] = {
    "KNN_50": knn_regressor,
    "RF": random_forest_regressor,
    "XGB": xgboost_poisson,
    "RF_CLS": random_forest_classifier,
    "TCN": tcn_regressor,
    "TCN_U": tcn_regressor_u,
    "TCN_CLS": tcn_classifier,
    "TCN_CLS_FL": tcn_classifier_fl,
    "TCN_CLS_WL": tcn_classifier_wl,
    "TCN_CLS_U_WL": tcn_classifier_u_wl,
    "TCN_CLS_U": tcn_classifier_u,
}
