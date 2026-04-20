import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .calibration import (
    apply_isotonic,
    fit_platt_intercept_slope,
    isotonic_calibrate,
    recalibrate_intercept_slope,
)
from .features import drop_leakage, make_preprocessor, split_columns
from .io import apply_column_mapping, load_column_mapping, read_table
from .models import build_elasticnet, build_ebm, build_xgb
from .run import _auto_feature_columns, _build_outcomes, _resolve_columns
from .utils import ensure_dir, write_json


@dataclass(frozen=True)
class ModelPick:
    outcome: str
    featureset: str
    model: str


def load_picks_tsv(path: Path) -> list[ModelPick]:
    df = pd.read_csv(path, sep="\t")
    need = {"outcome", "featureset", "model"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: {sorted(need)}")
    return [ModelPick(outcome=r["outcome"], featureset=r["featureset"], model=r["model"]) for _, r in df.iterrows()]


def _space_size(param_dist):
    size = 1
    for v in param_dist.values():
        size *= len(v) if isinstance(v, (list, tuple, np.ndarray)) else 1
    return size


def _search_best_estimator(pipeline, param_grid, cv, n_iter, seed, n_jobs):
    raise RuntimeError("Use _search_best_estimator_xy")


def _search_best_estimator_xy(pipeline, param_grid, X, y, cv, n_iter, seed, n_jobs):
    if not param_grid:
        pipeline.fit(X, y)
        return pipeline, {}
    total_space = _space_size(param_grid)
    if total_space > 0 and n_iter >= total_space:
        search = GridSearchCV(pipeline, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=n_jobs, error_score="raise")
    else:
        n_iter_eff = min(n_iter, total_space) if total_space > 0 else n_iter
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=n_iter_eff,
            scoring="roc_auc",
            cv=cv,
            random_state=seed,
            n_jobs=n_jobs,
            error_score="raise",
        )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def fit_models_from_picks(specs: dict, project_root: Path, picks_path: Path, out_dir: Path, tag: str = "primary"):
    """
    Train final (deployable) models on full labeled data, based on a picks TSV.

    This does NOT use the outer CV folds; it fits a single best estimator per pick using inner CV search,
    then optionally fits a calibration model (intercept_slope or isotonic) using OOF probs from inner CV.
    """
    validation = specs["validation"]
    model_spec = specs["model"]
    features_spec = specs["features"]
    data = specs["data"]

    mapping_path = (validation or {}).get("external_mapping_path")
    mapping = load_column_mapping(mapping_path) if mapping_path else {}

    df = read_table(
        data["path"],
        data["type"],
        encoding=data.get("encoding", "utf-8"),
        sheet_name=data.get("sheet_name", 0),
        missing_values=data.get("missing_values", []),
    )
    df = apply_column_mapping(df, mapping)

    outcomes = _build_outcomes(df, specs["outcomes"])

    feature_sets = dict(features_spec["feature_sets"])
    if features_spec.get("use_all_columns", False):
        target_sets = features_spec.get("use_all_columns_sets", [])
        if target_sets:
            feature_sets.update(_auto_feature_columns(df, specs, target_sets))

    leakage_block = features_spec.get("leakage_block", {})
    cat_cfg = features_spec["categorical"]
    scale_numeric = features_spec["scaling"].get("standardize_numeric", True)

    cv_inner = int(model_spec["cv"]["inner_folds"])
    seed = int(model_spec["cv"]["random_seed"])
    n_iter = int(model_spec["cv"]["n_iter"])
    search_n_jobs = int(model_spec["cv"].get("search_n_jobs", model_spec["cv"].get("n_jobs", 1)))
    model_n_jobs = int(model_spec["cv"].get("model_n_jobs", model_spec["cv"].get("n_jobs", 1)))
    inner = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=seed)

    recal_method = ((validation or {}).get("recalibration") or {}).get("method", "none")
    if recal_method in ("none", "off", "null", None):
        recal_method = "none"

    picks = load_picks_tsv(picks_path)
    for pick in picks:
        if pick.outcome not in outcomes:
            raise KeyError(f"Outcome not found: {pick.outcome}")
        if pick.featureset not in feature_sets:
            raise KeyError(f"Featureset not found: {pick.featureset}")

        y = pd.Series(outcomes[pick.outcome]).dropna().astype(int)
        outcome_base = pick.outcome.split("__", 1)[0]

        cols = _resolve_columns(df, feature_sets[pick.featureset])
        leakage_cols = _resolve_columns(df, leakage_block.get(outcome_base, []))
        cols = drop_leakage(cols, leakage_cols)

        X = df[cols].copy().loc[y.index]
        X_use, cat_cols, num_cols = split_columns(X, cols)
        pre = make_preprocessor(cat_cols, num_cols, cat_cfg, scale_numeric)

        if pick.model == "elasticnet":
            en_spec = model_spec["models"]["elasticnet"]
            model = build_elasticnet(seed, en_spec, n_jobs=model_n_jobs)
            pipe = Pipeline([("preprocess", pre), ("model", model)])
            params = {"model__C": en_spec["C"], "model__l1_ratio": en_spec["l1_ratio"]}
        elif pick.model == "xgboost":
            xgb_spec = model_spec["models"]["xgboost"]
            model = build_xgb(seed, xgb_spec, n_jobs=model_n_jobs)
            pipe = Pipeline([("preprocess", pre), ("model", model)])
            params = {
                "model__n_estimators": xgb_spec["n_estimators"],
                "model__max_depth": xgb_spec["max_depth"],
                "model__learning_rate": xgb_spec["learning_rate"],
                "model__subsample": xgb_spec["subsample"],
                "model__colsample_bytree": xgb_spec["colsample_bytree"],
            }
        elif pick.model == "ebm":
            ebm_spec = model_spec["models"].get("ebm", {})
            model = build_ebm(seed, ebm_spec)
            if model is None:
                raise RuntimeError("EBM requested but interpret is not available in this environment.")
            pipe = Pipeline([("preprocess", pre), ("model", model)])
            params = {
                "model__max_bins": ebm_spec.get("max_bins", [128]),
                "model__interactions": ebm_spec.get("interactions", [0]),
                "model__learning_rate": ebm_spec.get("learning_rate", [0.05]),
            }
        else:
            raise ValueError(f"Unsupported model for fit: {pick.model}")

        # Search best estimator on full data with inner CV
        best_estimator, best_params = _search_best_estimator_xy(
            pipe, params, X_use, y, inner, n_iter, seed, search_n_jobs
        )

        # Calibrator fit using inner-CV out-of-fold probs
        calibrator = {"method": recal_method}
        if recal_method != "none":
            p_oof = cross_val_predict(
                best_estimator, X_use, y, cv=inner, method="predict_proba", n_jobs=search_n_jobs
            )[:, 1]
            if recal_method == "intercept_slope":
                intercept, slope = fit_platt_intercept_slope(y, p_oof)
                calibrator.update({"intercept": float(intercept), "slope": float(slope)})
            elif recal_method == "isotonic":
                iso = isotonic_calibrate(y, p_oof)
                calibrator_path = out_dir / tag / pick.outcome / pick.featureset / pick.model / "calibrator_isotonic.joblib"
                ensure_dir(calibrator_path.parent)
                import joblib

                joblib.dump(iso, calibrator_path)
                calibrator.update({"isotonic_path": str(calibrator_path)})
            else:
                raise ValueError(f"Unsupported recalibration method: {recal_method}")

        # Save
        model_dir = out_dir / tag / pick.outcome / pick.featureset / pick.model
        ensure_dir(model_dir)
        import joblib

        joblib.dump(best_estimator, model_dir / "pipeline.joblib")
        write_json(model_dir / "best_params.json", {"best_params": best_params})
        write_json(model_dir / "calibrator.json", calibrator)
        write_json(
            model_dir / "meta.json",
            {
                "tag": tag,
                "outcome": pick.outcome,
                "featureset": pick.featureset,
                "model": pick.model,
                "n_train": int(len(y)),
                "pos": int(y.sum()),
                "pos_rate": float(y.mean()),
                "columns_used": cols,
                "leakage_blocked": leakage_cols,
            },
        )


def _load_pipeline(model_dir: Path):
    import joblib

    return joblib.load(model_dir / "pipeline.joblib")


def _load_calibrator(model_dir: Path):
    cal = json.loads((model_dir / "calibrator.json").read_text(encoding="utf-8"))
    if cal.get("method") == "isotonic":
        import joblib

        iso_path = cal.get("isotonic_path")
        if iso_path:
            cal["isotonic_model"] = joblib.load(iso_path)
    return cal


def predict_from_models(
    specs: dict,
    project_root: Path,
    models_root: Path,
    input_path: Path,
    output_path: Path,
    mapping_path: Path | None = None,
    include_tiers: bool = True,
    tier_thresholds: dict | None = None,
):
    """
    Apply trained models to a new Excel. Works for inputs with or without diagnosis columns.
    Outputs both raw and calibrated probabilities if calibrator exists.
    """
    validation = specs["validation"]
    data = specs["data"]

    mapping_path = mapping_path or (validation or {}).get("external_mapping_path")
    mapping = load_column_mapping(mapping_path) if mapping_path else {}

    df = read_table(
        str(input_path),
        "xlsx" if input_path.suffix.lower() in {".xlsx", ".xls"} else data.get("type", "xlsx"),
        encoding=data.get("encoding", "utf-8"),
        sheet_name=data.get("sheet_name", 0),
        missing_values=data.get("missing_values", []),
    )
    df = apply_column_mapping(df, mapping)

    rows = []
    for tag_dir in sorted(models_root.glob("*")):
        if not tag_dir.is_dir():
            continue
        tag = tag_dir.name
        for outcome_dir in sorted(tag_dir.glob("*")):
            outcome = outcome_dir.name
            for fs_dir in sorted(outcome_dir.glob("*")):
                featureset = fs_dir.name
                for model_dir in sorted(fs_dir.glob("*")):
                    if not (model_dir / "pipeline.joblib").exists():
                        continue
                    model_name = model_dir.name

                    pipeline = _load_pipeline(model_dir)
                    calibrator = _load_calibrator(model_dir)

                    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
                    cols = meta.get("columns_used") or []

                    X = df.copy()
                    # ensure required columns exist
                    for c in cols:
                        if c not in X.columns:
                            X[c] = np.nan
                    X = X[cols]

                    prob = pipeline.predict_proba(X)[:, 1]
                    prob_cal = None
                    if calibrator.get("method") == "intercept_slope" and "intercept" in calibrator and "slope" in calibrator:
                        prob_cal = recalibrate_intercept_slope(prob, calibrator["intercept"], calibrator["slope"])
                    elif calibrator.get("method") == "isotonic" and calibrator.get("isotonic_model") is not None:
                        prob_cal = apply_isotonic(calibrator["isotonic_model"], prob)

                    rows.append(
                        pd.DataFrame(
                            {
                                "row_index": np.arange(len(df)),
                                "tag": tag,
                                "outcome": outcome,
                                "featureset": featureset,
                                "model": model_name,
                                "prob": prob,
                                "prob_calibrated": prob_cal,
                            }
                        )
                    )

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # wide outputs for convenience
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")

    if include_tiers and not out.empty:
        thr = tier_thresholds or {"htn_hi": 0.8, "hyper_hi": 0.7, "lip_hi": 0.7, "any_mid": 0.6}
        # Build a wide table per tag using raw and calibrated
        for prob_variant in ["raw", "calibrated"]:
            pcol = "prob" if prob_variant == "raw" else "prob_calibrated"
            if prob_variant == "calibrated" and out[pcol].isna().all():
                continue

            wide = out.pivot_table(
                index=["row_index"],
                columns=["tag", "outcome"],
                values=pcol,
                aggfunc="first",
            )
            wide.columns = [f"{tag}__{outcome}__{pcol}" for tag, outcome in wide.columns]
            wide = wide.reset_index()

            # compute tiers for each tag separately
            for tag in sorted(out["tag"].unique()):
                h = wide.get(f"{tag}__hypertension__{pcol}")
                g = wide.get(f"{tag}__hyperglycemia__{pcol}")
                l = wide.get(f"{tag}__dyslipidemia__{pcol}")
                if h is None or g is None or l is None:
                    continue
                tier3 = (h >= thr["htn_hi"]) & (g >= thr["hyper_hi"])
                tier2 = (~tier3) & ((h >= thr["htn_hi"]) | (g >= thr["hyper_hi"]) | (l >= thr["lip_hi"]))
                tier1 = (~tier3) & (~tier2) & ((h >= thr["any_mid"]) | (g >= thr["any_mid"]) | (l >= thr["any_mid"]))
                tier = pd.Series("T0", index=wide.index)
                tier[tier1] = "T1"
                tier[tier2] = "T2"
                tier[tier3] = "T3"
                wide[f"{tag}__tier__{prob_variant}"] = tier

            wide.to_csv(output_path.with_suffix(f".{prob_variant}.wide.tsv"), sep="\t", index=False, encoding="utf-8-sig")
