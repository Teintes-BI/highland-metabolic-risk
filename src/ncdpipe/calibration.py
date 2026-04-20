import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def recalibrate_intercept_slope(prob, intercept, slope):
    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    new_logit = intercept + slope * logit
    return 1 / (1 + np.exp(-new_logit))


def fit_intercept_slope(y_true, prob):
    """
    Legacy linear fit (least squares) used previously as a calibration *diagnostic*.
    Prefer `fit_platt_intercept_slope` for a standard Platt-style logistic calibration model.
    """
    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    X = np.vstack([np.ones_like(logit), logit]).T
    coef = np.linalg.lstsq(X, y_true, rcond=None)[0]
    return float(coef[0]), float(coef[1])


def isotonic_calibrate(y_true, prob):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(prob, y_true)
    return iso


def fit_platt_intercept_slope(y_true, prob):
    """
    Standard intercept+slope recalibration (Platt-style):
      logit(p_cal) = intercept + slope * logit(p_raw)
    Fit by logistic regression on out-of-fold probabilities from the *training* portion.
    """
    eps = 1e-6
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(prob).astype(float), eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    # large C approximates "no regularization" while staying compatible across sklearn versions
    lr = LogisticRegression(solver="lbfgs", penalty="l2", C=1e6, max_iter=2000)
    lr.fit(logit, y)
    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0, 0])
    return intercept, slope


def apply_isotonic(iso: IsotonicRegression, prob):
    p = np.asarray(prob).astype(float)
    return iso.predict(p)


def binned_calibration_table(y_true, prob, bins=10):
    """
    Returns a calibration "curve" table using quantile bins:
      bin, n, pos, pos_rate, prob_mean, prob_median
    """
    df = pd.DataFrame({"y_true": y_true, "prob": prob}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["bin", "n", "pos", "pos_rate", "prob_mean", "prob_median"])
    df = df.sort_values("prob")
    df["bin"] = pd.qcut(df["prob"], q=bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True)
    out = (
        grouped.apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "pos": float(g["y_true"].sum()),
                    "pos_rate": float(g["y_true"].mean()),
                    "prob_mean": float(g["prob"].mean()),
                    "prob_median": float(g["prob"].median()),
                }
            )
        )
        .reset_index()
    )
    return out


def expected_calibration_error(binned_table: pd.DataFrame) -> float:
    """ECE = sum_k (n_k/N) * |acc_k - conf_k| using binned calibration table."""
    if binned_table is None or binned_table.empty:
        return float("nan")
    n = float(binned_table["n"].sum())
    if n <= 0:
        return float("nan")
    ece = ((binned_table["n"] / n) * (binned_table["pos_rate"] - binned_table["prob_mean"]).abs()).sum()
    return float(ece)
