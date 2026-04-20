import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=0.01, other_label="Other"):
        self.min_freq = min_freq
        self.other_label = other_label
        self.keep_map_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.keep_map_ = {}
        for col in X_df.columns:
            freq = X_df[col].value_counts(normalize=True, dropna=False)
            keep = set(freq[freq >= self.min_freq].index.tolist())
            self.keep_map_[col] = keep
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            keep = self.keep_map_.get(col, set())
            X_df[col] = X_df[col].where(X_df[col].isin(keep), self.other_label)
        return X_df


def split_columns(df, feature_list):
    X = df[feature_list].copy()
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, cat_cols, num_cols


def drop_leakage(feature_list, leakage_block):
    return [c for c in feature_list if c not in set(leakage_block)]


def make_preprocessor(cat_cols, num_cols, categorical_cfg, scale_numeric=True, use_splines=False, spline_cfg=None):
    min_freq = categorical_cfg.get("rare_level_threshold", 0.01)
    other_label = categorical_cfg.get("other_label", "Other")
    drop_first = categorical_cfg.get("drop_first", False)

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first" if drop_first else None)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, drop="first" if drop_first else None)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("rare", RareCategoryGrouper(min_freq=min_freq, other_label=other_label)),
        ("onehot", ohe)
    ])

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_splines:
        cfg = spline_cfg or {}
        n_knots = cfg.get("n_splines", 10)
        if isinstance(n_knots, (list, tuple)):
            n_knots = max(n_knots)
        n_knots = int(n_knots)
        num_steps.append(("spline", SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)))
    elif scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])


def get_feature_sets(features_spec):
    return features_spec["feature_sets"]
