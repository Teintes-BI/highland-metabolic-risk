import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _add_src_to_path(project_root: Path) -> None:
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _auto_feature_columns(df: pd.DataFrame, specs: dict, target_sets: list[str]) -> dict[str, list[str]]:
    # import from ncdpipe.run to stay consistent with pipeline
    from ncdpipe.run import _auto_feature_columns as _impl

    return _impl(df, specs, target_sets)


def _resolve_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    from ncdpipe.run import _resolve_columns as _impl

    return _impl(df, cols)


def _build_outcomes(df: pd.DataFrame, outcomes_spec: dict) -> dict[str, pd.Series]:
    from ncdpipe.run import _build_outcomes as _impl

    return _impl(df, outcomes_spec)


def main(args) -> int:
    project_root = Path(args.project_root).resolve()
    _add_src_to_path(project_root)

    from ncdpipe.config import load_specs
    from ncdpipe.io import read_table
    from ncdpipe.features import drop_leakage
    from ncdpipe.models import build_ebm

    specs = load_specs(project_root / "specs")
    data_spec = specs["data"]
    df = read_table(
        data_spec["path"],
        data_spec["type"],
        encoding=data_spec.get("encoding", "utf-8"),
        sheet_name=data_spec.get("sheet_name", 0),
        missing_values=data_spec.get("missing_values", []),
    )

    outcomes = _build_outcomes(df, specs["outcomes"])
    if args.outcome not in outcomes:
        raise KeyError(f"Outcome not found: {args.outcome}. Available: {sorted(outcomes.keys())}")

    y = pd.Series(outcomes[args.outcome]).dropna().astype(int)
    if y.nunique() < 2:
        raise ValueError(f"Outcome {args.outcome} has <2 classes after dropping NaNs.")

    features_spec = specs["features"]
    feature_sets = dict(features_spec["feature_sets"])
    if features_spec.get("use_all_columns", False):
        target_sets = features_spec.get("use_all_columns_sets", [])
        if target_sets:
            feature_sets.update(_auto_feature_columns(df, specs, target_sets))

    if args.featureset not in feature_sets:
        raise KeyError(f"Featureset not found: {args.featureset}. Available: {sorted(feature_sets.keys())}")

    cols_raw = feature_sets[args.featureset]
    cols = _resolve_columns(df, cols_raw) if cols_raw else []
    if not cols:
        # ALL_NONLEAK is typically empty but auto-filled above; if still empty, it's a configuration error
        raise ValueError(f"Featureset {args.featureset} resolved to 0 columns.")

    outcome_base = args.outcome.split("__", 1)[0]
    leakage_block = features_spec.get("leakage_block", {})
    leakage_cols = _resolve_columns(df, leakage_block.get(outcome_base, []))
    cols = drop_leakage(cols, leakage_cols)

    X = df[cols].copy().loc[y.index]

    # Prepare X for EBM: keep raw semantics (avoid one-hot) for better narrative plots.
    X_prepared = pd.DataFrame(index=X.index)
    feature_types: list[str] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            X_prepared[col] = pd.to_numeric(s, errors="coerce")
            feature_types.append("continuous")
            numeric_cols.append(col)
            continue

        # heuristic: try numeric conversion; if mostly numeric and reasonably high cardinality, treat as continuous
        s_num = pd.to_numeric(s, errors="coerce")
        non_na_ratio = float(s_num.notna().mean())
        uniq = int(s.dropna().nunique())
        if non_na_ratio >= args.numeric_ratio and uniq >= args.numeric_min_unique:
            X_prepared[col] = s_num
            feature_types.append("continuous")
            numeric_cols.append(col)
        else:
            X_prepared[col] = s.astype("object").where(~s.isna(), other="Missing").astype(str)
            feature_types.append("categorical")
            categorical_cols.append(col)

    seed = int(specs["model"]["cv"].get("random_seed", 42))
    ebm_spec = specs["model"]["models"].get("ebm", {})
    ebm = build_ebm(seed, ebm_spec)
    if ebm is None:
        raise RuntimeError("EBM is not available (interpret package missing).")

    if args.interactions is not None:
        ebm.set_params(interactions=int(args.interactions))

    ebm.fit(X_prepared, y)
    exp = ebm.explain_global(name=f"{args.outcome}:{args.featureset}")

    out_dir = project_root / "outputs" / "interpret" / "ebm"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Feature importance table
    imp = getattr(ebm, "feature_importances_", None)
    if imp is not None and len(imp) == X_prepared.shape[1]:
        imp_df = pd.DataFrame(
            {
                "feature": list(X_prepared.columns),
                "feature_type": feature_types,
                "importance": [float(x) for x in imp],
            }
        ).sort_values("importance", ascending=False)
        imp_df.to_csv(out_dir / f"{args.outcome}__{args.featureset}__feature_importance.tsv", sep="\t", index=False, encoding="utf-8-sig")

    # Global explanation data (JSON)
    payload = {
        "outcome": args.outcome,
        "featureset": args.featureset,
        "n": int(len(X_prepared)),
        "pos": int(y.sum()),
        "pos_rate": float(y.mean()),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "ebm_params": ebm.get_params(),
        "explain_global": exp.data(),
    }
    (out_dir / f"{args.outcome}__{args.featureset}__global.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Wrote: {out_dir}/{args.outcome}__{args.featureset}__global.json")
    if (out_dir / f"{args.outcome}__{args.featureset}__feature_importance.tsv").exists():
        print(f"Wrote: {out_dir}/{args.outcome}__{args.featureset}__feature_importance.tsv")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit an EBM on raw features and export global explanation for narrative.")
    parser.add_argument("--project_root", default=".", help="Project root containing src/ and specs/")
    parser.add_argument("--outcome", default="hyperglycemia", help="Outcome key (e.g., hyperglycemia)")
    parser.add_argument("--featureset", default="Q_ONLY", help="Featureset key (e.g., Q_ONLY)")
    parser.add_argument(
        "--numeric_ratio",
        type=float,
        default=0.8,
        help="If non-null ratio after numeric coercion >= this and unique>=min_unique, treat as continuous",
    )
    parser.add_argument("--numeric_min_unique", type=int, default=20, help="Minimum unique values to treat coerced numeric as continuous")
    parser.add_argument("--interactions", type=int, default=0, help="Set EBM interactions (0 recommended for clean plots)")
    raise SystemExit(main(parser.parse_args()))
