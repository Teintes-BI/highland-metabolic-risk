import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import make_preprocessor


def _aggregate_onehot(feature_names, base_features):
    agg = {b: [] for b in base_features}
    for name in feature_names:
        for b in base_features:
            if name == b or name.startswith(b + "_"):
                agg[b].append(name)
                break
    return agg


def stability_selection(X, y, base_features, categorical_cfg, scale_numeric, spec, random_state=42):
    B = int(spec.get("bootstrap_B", 200))
    frac = float(spec.get("subsample_frac", 0.7))

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = make_preprocessor(cat_cols, num_cols, categorical_cfg, scale_numeric)
    max_iter = int(spec.get("max_iter", 5000))
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        class_weight="balanced",
        max_iter=max_iter,
        n_jobs=-1,
        random_state=random_state
    )
    pipe = Pipeline([
        ("preprocess", pre),
        ("model", model)
    ])

    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    selected = []
    signs = []

    for _ in range(B):
        idx = rng.choice(n, size=int(n * frac), replace=True)
        X_sub = X.iloc[idx]
        y_sub = y.iloc[idx]
        pipe.fit(X_sub, y_sub)

        # 取得编码后特征名；对不支持 get_feature_names_out 的步骤回退到 ColumnTransformer
        try:
            names = pipe.named_steps["preprocess"].get_feature_names_out()
        except Exception:
            names = np.array([f"f{i}" for i in range(pipe.named_steps["preprocess"].transform(X_sub).shape[1])])
        coef = pipe.named_steps["model"].coef_.ravel()

        agg_map = _aggregate_onehot(names, base_features)
        row_sel = {}
        row_sign = {}
        for base, feats in agg_map.items():
            if not feats:
                continue
            idxs = [np.where(names == f)[0][0] for f in feats]
            vals = coef[idxs]
            max_idx = idxs[int(np.argmax(np.abs(vals)))]
            row_sel[base] = float(np.abs(coef[max_idx]) > 1e-6)
            row_sign[base] = float(np.sign(coef[max_idx]))
        selected.append(row_sel)
        signs.append(row_sign)

    sel_df = pd.DataFrame(selected).fillna(0.0)
    sign_df = pd.DataFrame(signs).fillna(0.0)

    out = []
    for col in sel_df.columns:
        freq = sel_df[col].mean()
        sign_consistency = np.mean(np.sign(sign_df[col]) == np.sign(sign_df[col].mean()))
        out.append({
            "feature": col,
            "inclusion_freq": round(float(freq), 6),
            "sign_consistency": round(float(sign_consistency), 6)
        })

    return pd.DataFrame(out)
