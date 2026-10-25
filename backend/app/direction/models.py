"""Direction models. Each answers: what is the chance that price reaches
+distance before -distance from the next candle's open?

  * "ml":       gradient-boosted decision trees (scikit-learn HistGradientBoosting,
                the same family as LightGBM) on the tabular features.
  * "dl":       an LSTM neural network (PyTorch) that reads the last 24 hours of
                30-minute candles in order, plus the tabular features as context.
  * "ensemble": the average of the two.
All are deliberately small and regularised: market data is noisy and a large
model would mostly memorise the past.
"""
from __future__ import annotations

import io
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import torch
    from torch import nn
    TORCH = True
except Exception:   # PyTorch not installed: the deep-learning model is skipped
    TORCH = False


class TreeModel:
    name = "ml"

    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.04, max_iter=250, min_samples_leaf=60,
            l2_regularization=2.0, random_state=0)
        self.empty_cols: list[str] = []

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.empty_cols:
            return X
        X = X.copy()
        X[self.empty_cols] = 0.0
        return X

    def fit(self, X: pd.DataFrame, seq, y) -> "TreeModel":
        self.empty_cols = [col for col in X.columns if X[col].isna().all()]
        self.model.fit(self._prep(X), y)
        return self

    def predict(self, X: pd.DataFrame, seq) -> np.ndarray:
        return self.model.predict_proba(self._prep(X))[:, 1]


if TORCH:
    class _Net(nn.Module):
        def __init__(self, n_static: int, n_seq: int, hidden: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(n_seq, hidden, batch_first=True)
            self.static = nn.Sequential(nn.Linear(n_static, hidden), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(hidden, 1))

        def forward(self, xs, xq):
            _, (hn, _) = self.lstm(xq)
            return self.head(torch.cat([hn[-1], self.static(xs)], dim=1)).squeeze(1)


class LSTMModel:
    name = "dl"

    def __init__(self, epochs: int = 25, batch: int = 256, lr: float = 1e-3, patience: int = 4, seed: int = 0):
        if not TORCH:
            raise RuntimeError("PyTorch is not installed")
        self.epochs, self.batch, self.lr, self.patience, self.seed = epochs, batch, lr, patience, seed
        self.mu = self.sd = None
        self.net = None

    def _static(self, X: pd.DataFrame) -> np.ndarray:
        a = (X.to_numpy(np.float32) - self.mu) / self.sd
        return np.clip(np.nan_to_num(a, nan=0.0), -6, 6).astype(np.float32)

    def fit(self, X: pd.DataFrame, seq: np.ndarray, y) -> "LSTMModel":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        A = X.to_numpy(np.float32)
        with warnings.catch_warnings():          # columns that are entirely empty (e.g. no funding data)
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mu = np.nanmean(A, axis=0)
            self.sd = np.nanstd(A, axis=0)
        self.mu = np.nan_to_num(self.mu, nan=0.0)
        self.sd = np.where(np.nan_to_num(self.sd, nan=0.0) > 1e-9, self.sd, 1.0).astype(np.float32)
        xs, xq, yy = self._static(X), np.asarray(seq, np.float32), np.asarray(y, np.float32)
        # last 15% of the (time-ordered) training rows decide when to stop training
        n_val = max(1, int(0.15 * len(yy)))
        tr, va = slice(0, len(yy) - n_val), slice(len(yy) - n_val, len(yy))
        self.net = _Net(xs.shape[1], xq.shape[2])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        Xs, Xq, Y = torch.from_numpy(xs), torch.from_numpy(xq), torch.from_numpy(yy)
        best, best_state, bad = np.inf, None, 0
        idx = np.arange(tr.stop)
        for _ in range(self.epochs):
            self.net.train()
            np.random.shuffle(idx)
            for k in range(0, len(idx), self.batch):
                b = torch.from_numpy(idx[k:k + self.batch])
                opt.zero_grad()
                loss = loss_fn(self.net(Xs[b], Xq[b]), Y[b])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = loss_fn(self.net(Xs[va], Xq[va]), Y[va]).item()
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X: pd.DataFrame, seq: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            z = self.net(torch.from_numpy(self._static(X)), torch.from_numpy(np.asarray(seq, np.float32)))
            return torch.sigmoid(z).numpy().astype(float)

    # saving / loading (state dict + scaler; no pickled code)
    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        torch.save({"state": self.net.state_dict(), "mu": self.mu, "sd": self.sd,
                    "n_static": len(self.mu), "n_seq": self.net.lstm.input_size}, buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "LSTMModel":
        d = torch.load(io.BytesIO(data), weights_only=False)
        m = cls()
        m.mu, m.sd = d["mu"], d["sd"]
        m.net = _Net(d["n_static"], d["n_seq"])
        m.net.load_state_dict(d["state"])
        m.net.eval()
        return m


def model_names(use_dl: bool = True) -> list[str]:
    return ["ml", "dl", "ensemble"] if (use_dl and TORCH) else ["ml"]


def make(name: str):
    return TreeModel() if name == "ml" else LSTMModel()
