import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from .io import apply_column_mapping, load_column_mapping, read_table
from .qc import run_qc
from .features import split_columns, make_preprocessor, drop_leakage
from .models import build_elasticnet, build_ebm, build_gam, build_xgb, compute_metrics, nested_cv_evaluate
from .calibration import binned_calibration_table, expected_calibration_error
from .stability import stability_selection
from .orthogonal import orthogonal_tables
from .report import write_report
from .utils import ensure_dir, set_seed, timestamp_run_id, write_json


def _normalize_col(name: str) -> str:
    """Normalize column names: strip spaces/punctuation, unify brackets/slashes, lowercase."""
    s = str(name).strip()
    repl = {
        "（": "(", "）": ")", "／": "/", "－": "-", "·": "",
        "²": "2", "、": "", "，": "", ",": "", "：": ":",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    s = "".join(s.split())
    return s.lower()


def _resolve_column(df: pd.DataFrame, col_name: str) -> str:
    col_map = {_normalize_col(c): c for c in df.columns}
    key = _normalize_col(col_name)
    if key not in col_map:
        import difflib
        candidates = difflib.get_close_matches(key, list(col_map.keys()), n=5, cutoff=0.6)
        raise KeyError(f"Column not found: {col_name}; normalized key={key}; close matches={candidates}")
    return col_map[key]


def _resolve_columns(df: pd.DataFrame, cols):
    return [_resolve_column(df, c) for c in cols]


def _build_outcomes(df: pd.DataFrame, outcomes_spec):
    def _to_float_bool(s):
        y = s.astype(float)
        return y

    def _eval_threshold(s, operator, thr):
        if operator in (">=", "ge", None):
            return s >= thr
        if operator in (">", "gt"):
            return s > thr
        if operator in ("<=", "le"):
            return s <= thr
        if operator in ("<", "lt"):
            return s < thr
        if operator in ("==", "eq"):
            return s == thr
        if operator in ("!=", "ne"):
            return s != thr
        raise ValueError(f"Unsupported operator: {operator}")

    def _eval_any_true(df, columns, positive_values=None, numeric_positive_as_true=True, missing_as_false=False):
        if not columns:
            return pd.Series(np.nan, index=df.index)
        cols = _resolve_columns(df, columns)
        sub = df[cols]
        miss = sub.isna().all(axis=1)
        pos_vals = set(positive_values or [1, "1", "是", "YES", "Yes", "yes", True])

        def _cell_true(v):
            if pd.isna(v):
                return False
            if isinstance(v, (int, float, np.integer, np.floating)) and numeric_positive_as_true:
                return float(v) > 0
            return str(v).strip() in set([str(x).strip() for x in pos_vals])

        # avoid deprecated DataFrame.applymap
        hit = sub.apply(lambda col: col.map(_cell_true)).any(axis=1)
        y = hit.astype(float)
        if missing_as_false:
            y[miss] = 0.0
        else:
            y[miss] = np.nan
        return y

    def _eval_rule(rule):
        kind = rule["kind"]
        if kind == "sbp_dbp":
            sbp_col = _resolve_column(df, rule["sbp_column"])
            dbp_col = _resolve_column(df, rule["dbp_column"])
            sbp = df[sbp_col]
            dbp = df[dbp_col]
            miss = sbp.isna() | dbp.isna()
            y = ((sbp >= rule["sbp_threshold"]) | (dbp >= rule["dbp_threshold"])).astype(float)
            y[miss] = np.nan
            return y
        if kind == "threshold":
            col = _resolve_column(df, rule["column"])
            s = df[col]
            op = rule.get("operator", ">=")
            thr = rule["threshold"]
            miss = s.isna()
            y = _eval_threshold(s, op, thr).astype(float)
            y[miss] = np.nan
            return y
        if kind == "lipid":
            tg = df[_resolve_column(df, rule["tg_column"])]
            tc = df[_resolve_column(df, rule["tc_column"])]
            ldl = df[_resolve_column(df, rule["ldl_column"])]
            hdl = df[_resolve_column(df, rule["hdl_column"])]
            sex_raw = df[_resolve_column(df, rule["sex_column"])]
            sex = sex_raw.astype(str)
            male_label = rule.get("sex_male_value", "男")
            female_label = rule.get("sex_female_value", "女")
            miss = tg.isna() | tc.isna() | ldl.isna() | hdl.isna() | sex_raw.isna()
            hdl_low = ((sex == male_label) & (hdl < rule["hdl_male_threshold"])) | (
                (sex == female_label) & (hdl < rule["hdl_female_threshold"])
            )
            y = ((tg >= rule["tg_threshold"]) | (tc >= rule["tc_threshold"]) | (ldl >= rule["ldl_threshold"]) | hdl_low).astype(float)
            y[miss] = np.nan
            return y
        if kind == "hdl_low":
            hdl = df[_resolve_column(df, rule["hdl_column"])]
            sex_raw = df[_resolve_column(df, rule["sex_column"])]
            sex = sex_raw.astype(str)
            male_label = rule.get("sex_male_value", "男")
            female_label = rule.get("sex_female_value", "女")
            miss = hdl.isna() | sex_raw.isna()
            hdl_low = ((sex == male_label) & (hdl < rule["hdl_male_threshold"])) | (
                (sex == female_label) & (hdl < rule["hdl_female_threshold"])
            )
            y = hdl_low.astype(float)
            y[miss] = np.nan
            return y
        if kind == "waist_by_sex":
            waist = df[_resolve_column(df, rule["waist_column"])]
            sex_raw = df[_resolve_column(df, rule["sex_column"])]
            sex = sex_raw.astype(str)
            male_label = rule.get("sex_male_value", "男")
            female_label = rule.get("sex_female_value", "女")
            valid_sex = sex.isin([male_label, female_label])
            miss = waist.isna() | sex_raw.isna() | (~valid_sex)
            male_thr = float(rule.get("male_threshold", 90))
            female_thr = float(rule.get("female_threshold", 85))
            y = (((sex == male_label) & (waist >= male_thr)) | ((sex == female_label) & (waist >= female_thr))).astype(float)
            y[miss] = np.nan
            return y
        if kind == "non_hdl":
            tc = df[_resolve_column(df, rule["tc_column"])]
            hdl = df[_resolve_column(df, rule["hdl_column"])]
            miss = tc.isna() | hdl.isna()
            non_hdl = tc - hdl
            thr = rule.get("threshold", 4.1)
            y = (non_hdl >= thr).astype(float)
            y[miss] = np.nan
            return y
        if kind == "aip":
            tg = df[_resolve_column(df, rule["tg_column"])]
            hdl = df[_resolve_column(df, rule["hdl_column"])]
            miss = tg.isna() | hdl.isna() | (hdl <= 0)
            val = np.log10(tg / hdl)
            thr = rule.get("threshold", 0.24)
            y = (val >= thr).astype(float)
            y[miss] = np.nan
            return y
        if kind == "tyg":
            tg = df[_resolve_column(df, rule["tg_column"])]
            glu = df[_resolve_column(df, rule["glucose_column"])]
            miss = tg.isna() | glu.isna() | (tg <= 0) | (glu <= 0)
            val = np.log(tg * glu / 2.0)
            thr = rule.get("threshold", 8.8)
            y = (val >= thr).astype(float)
            y[miss] = np.nan
            return y
        if kind == "count_positive":
            rules = rule.get("rules", [])
            min_count = int(rule.get("min_count", 2))
            if not rules:
                return pd.Series(np.nan, index=df.index)
            ys = [_eval_rule(r) for r in rules]
            ys = [pd.Series(y, index=df.index) for y in ys]
            # treat nan as 0 for counting
            stack = np.vstack([y.fillna(0).values for y in ys])
            counts = np.sum(stack >= 1, axis=0)
            y = (counts >= min_count).astype(float)
            # If all subrules nan for a row -> nan
            all_nan = np.all(np.vstack([y0.isna().values for y0 in ys]), axis=0)
            y[all_nan] = np.nan
            return pd.Series(y, index=df.index).astype(float)
        if kind == "any_true":
            return _eval_any_true(
                df,
                columns=rule.get("columns", []),
                positive_values=rule.get("positive_values"),
                numeric_positive_as_true=bool(rule.get("numeric_positive_as_true", True)),
                missing_as_false=bool(rule.get("missing_as_false", False)),
            )
        if kind == "any_of":
            rules = rule.get("rules", [])
            if not rules:
                return pd.Series(np.nan, index=df.index)
            ys = [_eval_rule(r) for r in rules]
            ys = [pd.Series(y, index=df.index) for y in ys]
            stack = np.vstack([y.values for y in ys])
            any_pos = np.nanmax(stack, axis=0)  # nan if all nan
            # nanmax returns nan only if all nan; but if some nan and some 0, it returns 0
            return pd.Series(any_pos, index=df.index).astype(float)
        if kind == "and_not":
            """
            Combine one positive rule with one or more exclusion rules.
            - include: a single rule dict (required)
            - exclude_rules: list of rule dicts (optional)
            Result = 1 only if include==1 and none of the exclude rules ==1.
            If include is nan -> nan. Exclusion nan are treated as 0 (do not exclude).
            """
            include_rule = rule.get("include")
            if not include_rule:
                return pd.Series(np.nan, index=df.index)
            inc = pd.Series(_eval_rule(include_rule), index=df.index)
            if "exclude_rules" in rule:
                ex_rules = rule.get("exclude_rules") or []
                if ex_rules:
                    ex_vals = []
                    for r in ex_rules:
                        ex = pd.Series(_eval_rule(r), index=df.index)
                        # treat nan as 0 for exclusion logic
                        ex_vals.append(ex.fillna(0.0))
                    if ex_vals:
                        ex_stack = np.vstack([x.values for x in ex_vals])
                        ex_any = np.nanmax(ex_stack, axis=0)
                    else:
                        ex_any = np.zeros(len(df))
                else:
                    ex_any = np.zeros(len(df))
            else:
                ex_any = np.zeros(len(df))
            out = pd.Series(0.0, index=df.index)
            out[inc.isna()] = np.nan
            out[(inc == 1) & (ex_any < 1)] = 1.0
            return out.astype(float)
        raise ValueError(f"Unsupported rule kind: {kind}")

    out = {}
    for outcome_name, spec in outcomes_spec.items():
        # 支持 primary/secondary 多口径：outcome__definition
        if "definitions" in spec:
            defs = spec["definitions"] or {}
            run_defs = spec.get("run_definitions") or list(defs.keys())
            # 保证 primary/secondary 顺序更稳定
            if isinstance(run_defs, (list, tuple)):
                run_defs = list(run_defs)
            else:
                run_defs = [str(run_defs)]
            for def_name in run_defs:
                if def_name not in defs:
                    raise KeyError(f"Outcome {outcome_name} missing definition: {def_name}")
                rule = defs[def_name]["rule"]
                out[f"{outcome_name}__{def_name}"] = _eval_rule(rule)
        else:
            rule = spec["rule"]
            out[outcome_name] = _eval_rule(rule)
    return out


def _auto_feature_columns(df: pd.DataFrame, specs, target_sets):
    data_spec = specs["data"]
    features_spec = specs["features"]
    exclude = set(features_spec.get("exclude_columns", []))
    if data_spec.get("id_column"):
        exclude.add(data_spec["id_column"])
    exclude.update(data_spec.get("pii_columns", []))
    # 保留实际列名，避免归一化误伤
    cols = [c for c in df.columns if c not in exclude]
    return {name: cols for name in target_sets}


def run_pipeline(specs, project_root: Path, run_id=None):
    data = specs["data"]
    features_spec = specs["features"]
    model_spec = specs["model"]
    validation = specs["validation"]
    recal_cfg = (validation or {}).get("recalibration", {}) or {}
    recal_method = (recal_cfg.get("method") or "none").strip()
    recalibration = None
    # Parallelism controls (search vs model internal threads)
    search_n_jobs_global = int(model_spec["cv"].get("search_n_jobs", model_spec["cv"].get("n_jobs", -1)))
    if recal_method not in ("none", "off", "null"):
        recalibration = {"method": recal_method, "n_jobs": search_n_jobs_global}
    calib_bins = int(((validation or {}).get("calibration") or {}).get("ece_bins", 10))

    run_id = run_id or timestamp_run_id()
    run_dir = project_root / "runs" / run_id
    ensure_dir(run_dir)
    ensure_dir(run_dir / "specs")

    set_seed(int(model_spec["cv"].get("random_seed", 42)))

    df = read_table(
        data["path"],
        data["type"],
        encoding=data.get("encoding", "utf-8"),
        sheet_name=data.get("sheet_name", 0),
        missing_values=data.get("missing_values", []),
    )
    mapping_path = (validation or {}).get("external_mapping_path")
    mapping = load_column_mapping(mapping_path) if mapping_path else {}
    df = apply_column_mapping(df, mapping)

    outcomes = _build_outcomes(df, specs["outcomes"])

    # QC
    run_qc(specs, project_root)
    _write_label_qc(outcomes, project_root)

    outputs_root = project_root / "outputs"
    ensure_dir(outputs_root)

    results = {
        "run_id": run_id,
        "outcomes": {},
    }

    feature_sets = dict(features_spec["feature_sets"])
    if features_spec.get("use_all_columns", False):
        target_sets = features_spec.get("use_all_columns_sets", [])
        if target_sets:
            auto_sets = _auto_feature_columns(df, specs, target_sets)
            feature_sets.update(auto_sets)
    cat_cfg = features_spec["categorical"]
    scale_numeric = features_spec["scaling"].get("standardize_numeric", True)
    leakage_block = features_spec.get("leakage_block", {})

    for outcome_name, y in outcomes.items():
        outcome_base = outcome_name.split("__", 1)[0]
        y_series = pd.Series(y).dropna()
        outcome_res = {}
        for fs_name, fs_cols in feature_sets.items():
            cols = _resolve_columns(df, fs_cols)
            cols = drop_leakage(cols, _resolve_columns(df, leakage_block.get(outcome_base, [])))
            X = df[cols].copy()
            X = X.loc[y_series.index]
            y_use = y_series.astype(int)

            X_use, cat_cols, num_cols = split_columns(X, cols)
            pre = make_preprocessor(cat_cols, num_cols, cat_cfg, scale_numeric)

            cv_outer = model_spec["cv"]["outer_folds"]
            cv_inner = model_spec["cv"]["inner_folds"]
            seed = int(model_spec["cv"]["random_seed"])
            n_iter = int(model_spec["cv"]["n_iter"])
            search_n_jobs = int(model_spec["cv"].get("search_n_jobs", model_spec["cv"].get("n_jobs", -1)))
            model_n_jobs = int(model_spec["cv"].get("model_n_jobs", model_spec["cv"].get("n_jobs", -1)))

            outer = StratifiedKFold(n_splits=cv_outer, shuffle=True, random_state=seed)
            inner = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=seed)

            models_out = {}

            # ElasticNet
            en_spec = model_spec["models"]["elasticnet"]
            en_model = build_elasticnet(seed, en_spec, n_jobs=model_n_jobs)
            en_pipe = Pipeline([("preprocess", pre), ("model", en_model)])
            en_params = {
                "model__C": en_spec["C"],
                "model__l1_ratio": en_spec["l1_ratio"],
            }
            rows, best, preds = nested_cv_evaluate(
                X_use, y_use, en_pipe, en_params, outer, inner, n_iter, seed, recalibration=recalibration, n_jobs=search_n_jobs
            )
            models_out["elasticnet"] = {"fold_metrics": rows, "best_params": best, "fold_predictions": preds}

            # EBM or GAM fallback
            ebm_spec = model_spec["models"].get("ebm", {})
            if ebm_spec.get("run", True):
                ebm_model = build_ebm(seed, ebm_spec)
                if ebm_model is not None:
                    ebm_pipe = Pipeline([("preprocess", pre), ("model", ebm_model)])
                    ebm_params = {
                        "model__max_bins": ebm_spec.get("max_bins", [128]),
                        "model__interactions": ebm_spec.get("interactions", [0]),
                        "model__learning_rate": ebm_spec.get("learning_rate", [0.05]),
                    }
                    rows, best, preds = nested_cv_evaluate(
                        X_use, y_use, ebm_pipe, ebm_params, outer, inner, n_iter, seed, recalibration=recalibration, n_jobs=search_n_jobs
                    )
                    models_out["ebm"] = {"fold_metrics": rows, "best_params": best, "fold_predictions": preds}
                else:
                    gam_spec = model_spec["models"].get("gam", {})
                    if gam_spec.get("run", True):
                        pre_gam = make_preprocessor(cat_cols, num_cols, cat_cfg, scale_numeric, use_splines=True, spline_cfg=gam_spec)
                        gam_model = build_gam(seed, gam_spec, n_jobs=model_n_jobs)
                        gam_pipe = Pipeline([("preprocess", pre_gam), ("model", gam_model)])
                        gam_params = {}
                        rows, best, preds = nested_cv_evaluate(
                            X_use, y_use, gam_pipe, gam_params, outer, inner, n_iter, seed, recalibration=recalibration, n_jobs=search_n_jobs
                        )
                        models_out["gam"] = {"fold_metrics": rows, "best_params": best, "fold_predictions": preds}

            # XGBoost / fallback
            xgb_spec = model_spec["models"].get("xgboost", {})
            if xgb_spec.get("run", True):
                xgb_model = build_xgb(seed, xgb_spec, n_jobs=model_n_jobs)
                xgb_pipe = Pipeline([("preprocess", pre), ("model", xgb_model)])
                xgb_params = {
                    "model__n_estimators": xgb_spec["n_estimators"],
                    "model__max_depth": xgb_spec["max_depth"],
                    "model__learning_rate": xgb_spec["learning_rate"],
                    "model__subsample": xgb_spec["subsample"],
                    "model__colsample_bytree": xgb_spec["colsample_bytree"],
                }
                rows, best, preds = nested_cv_evaluate(
                    X_use, y_use, xgb_pipe, xgb_params, outer, inner, n_iter, seed, recalibration=recalibration, n_jobs=search_n_jobs
                )
                models_out["xgboost"] = {"fold_metrics": rows, "best_params": best, "fold_predictions": preds}

            outcome_res[fs_name] = models_out

            out_dir = outputs_root / "models" / outcome_name / fs_name
            ensure_dir(out_dir)
            for model_name, res in models_out.items():
                mdir = out_dir / model_name
                ensure_dir(mdir)
                preds = res.pop("fold_predictions", None)
                pd.DataFrame(res["fold_metrics"]).to_csv(
                    mdir / "fold_metrics.tsv", sep="\t", index=False, encoding="utf-8-sig"
                )
                write_json(mdir / "best_params.json", {"best_params": res["best_params"]})
                if preds:
                    pred_df = pd.concat(preds, ignore_index=True)
                    pred_df.to_csv(mdir / "fold_predictions.tsv", sep="\t", index=False, encoding="utf-8-sig")

                    # Export calibration curves/tables (raw + calibrated if available)
                    cal_dir = outputs_root / "calibration" / outcome_name / fs_name / model_name
                    ensure_dir(cal_dir)
                    raw_bins = binned_calibration_table(pred_df["y_true"], pred_df["prob"], bins=calib_bins)
                    raw_bins.to_csv(cal_dir / "bins_raw.tsv", sep="\t", index=False, encoding="utf-8-sig")
                    raw_ece = expected_calibration_error(raw_bins)
                    raw_metrics = compute_metrics(pred_df["y_true"].values, pred_df["prob"].values)

                    cal_metrics_row = {
                        "outcome": outcome_name,
                        "featureset": fs_name,
                        "model": model_name,
                        "n": int(len(pred_df)),
                        "ece_bins": int(calib_bins),
                        "ece_raw": float(raw_ece),
                        "brier_raw": float(raw_metrics["brier"]),
                    }

                    if "prob_calibrated" in pred_df.columns and pred_df["prob_calibrated"].notna().any():
                        cal_bins = binned_calibration_table(
                            pred_df["y_true"], pred_df["prob_calibrated"], bins=calib_bins
                        )
                        cal_bins.to_csv(cal_dir / "bins_calibrated.tsv", sep="\t", index=False, encoding="utf-8-sig")
                        cal_ece = expected_calibration_error(cal_bins)
                        cal_metrics = compute_metrics(pred_df["y_true"].values, pred_df["prob_calibrated"].values)
                        cal_metrics_row.update(
                            {
                                "ece_calibrated": float(cal_ece),
                                "brier_calibrated": float(cal_metrics["brier"]),
                            }
                        )

                    pd.DataFrame([cal_metrics_row]).to_csv(
                        cal_dir / "calibration_metrics.tsv", sep="\t", index=False, encoding="utf-8-sig"
                    )

        results["outcomes"][outcome_name] = outcome_res

    summary_path = outputs_root / "summary.json"
    write_json(summary_path, results)
    write_json(run_dir / "summary.json", results)

    # Snapshot specs
    specs_dir = project_root / "specs"
    for item in specs_dir.glob("*.yaml"):
        shutil.copy2(item, run_dir / "specs" / item.name)

    # Stability selection (F2)
    stab_spec = specs["stability"]
    fs_name = stab_spec.get("featureset", "F2")
    base_cols_raw = feature_sets.get(fs_name, [])
    if base_cols_raw:
        base_cols = _resolve_columns(df, base_cols_raw)
        X_base = df[base_cols].copy()
        cat_cfg = features_spec["categorical"]
        scale_numeric = features_spec["scaling"].get("standardize_numeric", True)
        for outcome_name, y in outcomes.items():
            outcome_base = outcome_name.split("__", 1)[0]
            y_series = pd.Series(y).dropna().astype(int)
            leakage_cols = _resolve_columns(df, leakage_block.get(outcome_base, []))
            use_cols = drop_leakage(base_cols, leakage_cols)
            X_use = df[use_cols].copy().loc[y_series.index]
            stab_df = stability_selection(
                X_use,
                y_series,
                use_cols,
                cat_cfg,
                scale_numeric,
                stab_spec,
            )
            out_dir = outputs_root / "stability" / outcome_name
            ensure_dir(out_dir)
            stab_df.to_csv(out_dir / "stability_table.tsv", sep="\t", index=False, encoding="utf-8-sig")

    # Orthogonal tables (raw)
    ortho = specs["orthogonal"]
    for outcome_name, y in outcomes.items():
        y_series = pd.Series(y)
        tmp = df.copy()
        tmp["_outcome_"] = y_series
        binning_cfg = ortho.get("binning", {})
        binning_resolved = {}
        for k, v in binning_cfg.items():
            binning_resolved[_resolve_column(df, k)] = v
        for combo in ortho.get("combos", []):
            combo["variables"] = _resolve_columns(df, combo.get("variables", []))
        tables = orthogonal_tables(
            tmp,
            "_outcome_",
            ortho.get("combos", []),
            binning_resolved,
            min_cell_n=ortho.get("min_cell_n", 50),
            model=None,
        )
        out_dir = outputs_root / "orthogonal_tables" / outcome_name
        ensure_dir(out_dir)
        for name, table in tables.items():
            table.to_csv(out_dir / f"table_{name}.tsv", sep="\t", index=False, encoding="utf-8-sig")

    # Report
    report_dir = outputs_root / "report"
    ensure_dir(report_dir)
    write_report(report_dir, results)

    return summary_path


def _write_label_qc(outcomes: dict, project_root: Path):
    """Label consistency checks: primary vs secondary by outcome base."""
    qc_dir = project_root / "outputs" / "qc"
    ensure_dir(qc_dir)

    rows = []
    # collect base names
    bases = sorted({k.split("__", 1)[0] for k in outcomes.keys()})
    for base in bases:
        primary_key = f"{base}__primary"
        secondary_key = f"{base}__secondary"
        if primary_key not in outcomes or secondary_key not in outcomes:
            continue
        y_primary = pd.Series(outcomes[primary_key])
        y_secondary = pd.Series(outcomes[secondary_key])
        valid = ~(y_primary.isna() | y_secondary.isna())
        if valid.sum() == 0:
            continue
        p = y_primary[valid].astype(int)
        s = y_secondary[valid].astype(int)

        ctab = pd.crosstab(p, s, dropna=False)
        # ensure all cells exist
        for i in [0, 1]:
            for j in [0, 1]:
                if j not in ctab.columns or i not in ctab.index:
                    ctab.loc[i, j] = 0
        ctab = ctab.sort_index().sort_index(axis=1)
        n_total = int(valid.sum())
        n_pp = int(ctab.loc[1, 1])
        n_p1_s0 = int(ctab.loc[1, 0])
        n_p0_s1 = int(ctab.loc[0, 1])
        n_p0_s0 = int(ctab.loc[0, 0])

        rows.append({
            "outcome": base,
            "n_total": n_total,
            "primary_pos": int(p.sum()),
            "secondary_pos": int(s.sum()),
            "both_pos": n_pp,
            "primary_pos_secondary_neg": n_p1_s0,
            "secondary_pos_primary_neg": n_p0_s1,
            "primary_pos_secondary_neg_rate": round(n_p1_s0 / n_total, 6),
            "secondary_pos_primary_neg_rate": round(n_p0_s1 / n_total, 6),
        })

        ctab_out = pd.DataFrame({
            "primary": [0, 0, 1, 1],
            "secondary": [0, 1, 0, 1],
            "count": [n_p0_s0, n_p0_s1, n_p1_s0, n_pp],
            "rate": [round(n_p0_s0 / n_total, 6), round(n_p0_s1 / n_total, 6),
                     round(n_p1_s0 / n_total, 6), round(n_pp / n_total, 6)]
        })
        ctab_out.to_csv(qc_dir / f"label_crosstab_{base}.tsv", sep="\t", index=False, encoding="utf-8-sig")

    if rows:
        pd.DataFrame(rows).to_csv(qc_dir / "label_consistency.tsv", sep="\t", index=False, encoding="utf-8-sig")
