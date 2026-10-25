"""A model that estimates, for each ORB breakout, the chance that the trade
ends in profit (after costs), and a cut-off below which the trade is skipped.

Honest-evaluation recipe:
  1. Development period (first 70% of history) only:
       - Out-of-fold predictions with a time-series split: the model is always
         trained on EARLIER trades and predicts LATER ones it has never seen.
       - The cut-off is chosen on those out-of-fold predictions.
  2. A final model is trained on the whole development period.
  3. It predicts the test period (last 30%), which played no part in any choice.
The model is deliberately small and heavily regularised: there are only a
few hundred trades per year, and a flexible model would memorise noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


class FilterModel:
    """Small gradient-boosted tree classifier. Columns that are entirely empty
    in the training data (e.g. funding history unavailable) are set to 0, which
    the model then ignores; other missing values are handled natively."""

    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40,
            l2_regularization=1.0, random_state=0,
        )
        self.empty_cols: list[str] = []
        self.classes_ = np.array([0, 1])

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.empty_cols:
            return X
        X = X.copy()
        X[self.empty_cols] = 0.0
        return X

    def fit(self, X: pd.DataFrame, y) -> "FilterModel":
        self.empty_cols = [c for c in X.columns if X[c].isna().all()]
        self.model.fit(self._prep(X), y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._prep(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prep(X))



def make_model() -> FilterModel:
    return FilterModel()


def oof_predictions(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Probability of a win for each row, predicted by a model trained only on
    earlier rows. The first block (no earlier data) is NaN."""
    p = np.full(len(X), np.nan)
    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        if len(np.unique(y[train_idx])) < 2:
            continue
        m = make_model().fit(X.iloc[train_idx], y[train_idx])
        p[test_idx] = m.predict_proba(X.iloc[test_idx])[:, 1]
    return p


def choose_threshold(p: np.ndarray, r: np.ndarray, min_keep: int) -> tuple[float, dict]:
    """Cut-off on predicted win chance that maximises the average result per
    trade of the kept trades, keeping at least `min_keep` trades."""
    ok = ~np.isnan(p)
    p, r = p[ok], r[ok]
    best = (0.0, r.mean(), len(r))          # threshold 0 = take every trade
    for q in np.arange(0.0, 0.80, 0.05):
        thr = float(np.quantile(p, q))
        keep = p >= thr
        if keep.sum() < min_keep:
            break
        mean_r = r[keep].mean()
        if mean_r > best[1]:
            best = (thr, mean_r, int(keep.sum()))
    return best[0], {"kept": best[2], "of": len(r), "mean_r": best[1]}


def auc(y: np.ndarray, p: np.ndarray) -> float | None:
    ok = ~np.isnan(p)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return None
    return float(roc_auc_score(y[ok], p[ok]))


def permutation_importance_auc(model, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 10,
                               seed: int = 0) -> list[tuple[str, float]]:
    """Drop in AUC when one feature's values are shuffled (bigger = more useful)."""
    base = auc(y, model.predict_proba(X)[:, 1])
    if base is None:
        return []
    rng = np.random.default_rng(seed)
    out = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - auc(y, model.predict_proba(Xp)[:, 1]))
        out.append((col, float(np.mean(drops))))
    return sorted(out, key=lambda x: -x[1])
