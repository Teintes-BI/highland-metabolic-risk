import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.metrics import accuracy_score, f1_score
from functools import reduce

from .calibration import (
    apply_isotonic,
    fit_platt_intercept_slope,
    isotonic_calibrate,
    recalibrate_intercept_slope,
)

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_EBM = True
except Exception:
    HAS_EBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from sklearn.ensemble import GradientBoostingClassifier


def _first(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple, np.ndarray)):
        return value[0] if len(value) > 0 else default
    return value


def build_elasticnet(random_state, spec, n_jobs=-1):
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=spec.get("max_iter", 2000),
        class_weight=spec.get("class_weight", "balanced"),
        n_jobs=n_jobs,
        random_state=random_state
    )


def build_ebm(random_state, spec):
    if not HAS_EBM:
        return None
    return ExplainableBoostingClassifier(
        max_bins=int(_first(spec.get("max_bins", 128))),
        interactions=int(_first(spec.get("interactions", 0))),
        learning_rate=float(_first(spec.get("learning_rate", 0.05))),
        max_rounds=int(_first(spec.get("max_rounds", 500))),
        random_state=random_state
    )


def build_xgb(random_state, spec, n_jobs=-1):
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=int(_first(spec.get("n_estimators", 300))),
            max_depth=int(_first(spec.get("max_depth", 4))),
            learning_rate=float(_first(spec.get("learning_rate", 0.05))),
            subsample=float(_first(spec.get("subsample", 0.8))),
            colsample_bytree=float(_first(spec.get("colsample_bytree", 0.8))),
            eval_metric="logloss",
            n_jobs=n_jobs,
            random_state=random_state
        )
    return GradientBoostingClassifier(random_state=random_state)


def build_gam(random_state, spec, n_jobs=-1):
    return LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=random_state
    )


def _calibration_slope_intercept(y_true, prob):
    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    X = np.vstack([np.ones_like(logit), logit]).T
    coef = np.linalg.lstsq(X, y_true, rcond=None)[0]
    intercept = coef[0]
    slope = coef[1]
    return intercept, slope


def compute_metrics(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred)
    }
    intercept, slope = _calibration_slope_intercept(y_true, prob)
    metrics["calibration_intercept"] = float(intercept)
    metrics["calibration_slope"] = float(slope)
    return metrics


def _space_size(param_dist):
    size = 1
    for v in param_dist.values():
        if isinstance(v, (list, tuple, np.ndarray)):
            size *= len(v)
        else:
            size *= 1
    return size


def nested_cv_evaluate(
    X,
    y,
    pipeline,
    param_dist,
    cv_outer,
    cv_inner,
    n_iter,
    random_state,
    recalibration: dict | None = None,
    n_jobs: int = -1,
):
    rows = []
    best_params = []
    fold_predictions = []

    for fold, (train_idx, test_idx) in enumerate(cv_outer.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        used_estimator = pipeline
        if not param_dist:
            used_estimator.fit(X_train, y_train)
            prob = used_estimator.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, prob)
            metrics["fold"] = fold
            rows.append(metrics)
            best_params.append({})
        else:
            total_space = _space_size(param_dist)
            # 当参数组合较少时改用 GridSearch，避免 n_iter 提示并用尽搜索空间
            if total_space > 0 and n_iter >= total_space:
                search = GridSearchCV(
                    pipeline,
                    param_grid=param_dist,
                    scoring="roc_auc",
                    cv=cv_inner,
                    n_jobs=n_jobs,
                    error_score="raise"
                )
            else:
                n_iter_eff = min(n_iter, total_space) if total_space > 0 else n_iter
                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=param_dist,
                    n_iter=n_iter_eff,
                    scoring="roc_auc",
                    cv=cv_inner,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    error_score="raise"
                )
            search.fit(X_train, y_train)
            used_estimator = search.best_estimator_
            prob = used_estimator.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, prob)
            metrics["fold"] = fold
            rows.append(metrics)
            best_params.append(search.best_params_)

        # Optional CV-safe recalibration:
        # Fit calibrator on out-of-fold predictions from TRAIN portion only (using cv_inner),
        # then apply to TEST probabilities. This avoids calibrating on the same TEST fold.
        prob_calibrated = None
        if recalibration:
            method = recalibration.get("method", "intercept_slope")
            n_jobs = int(recalibration.get("n_jobs", 1))
            # training OOF probs for calibrator
            p_train_oof = cross_val_predict(
                used_estimator,
                X_train,
                y_train,
                cv=cv_inner,
                method="predict_proba",
                n_jobs=n_jobs,
            )[:, 1]
            if method == "intercept_slope":
                intercept, slope = fit_platt_intercept_slope(y_train, p_train_oof)
                prob_calibrated = recalibrate_intercept_slope(prob, intercept, slope)
            elif method == "isotonic":
                iso = isotonic_calibrate(y_train, p_train_oof)
                prob_calibrated = apply_isotonic(iso, prob)
            elif method in (None, "none", "off"):
                prob_calibrated = None
            else:
                raise ValueError(f"Unsupported recalibration method: {method}")

            if prob_calibrated is not None:
                metrics_cal = compute_metrics(y_test, prob_calibrated)
                # keep raw metrics as canonical; add calibrated suffixes for comparison
                for k, v in metrics_cal.items():
                    rows[-1][f"{k}_calibrated"] = float(v)

        fold_predictions.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "row_index": X_test.index,
                    "y_true": y_test.values,
                    "prob": prob,
                    "prob_calibrated": prob_calibrated,
                }
            )
        )

    return rows, best_params, fold_predictions
