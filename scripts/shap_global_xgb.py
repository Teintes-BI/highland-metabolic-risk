import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["font.family"] = 'Noto Sans CJK SC'  # 换成你有的中文字体

# Ensure local ncdpipe package is importable when loading pipeline.joblib
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_pipeline(model_dir: Path):
    import joblib

    return joblib.load(model_dir / "pipeline.joblib")


def _get_feature_names(preprocess):
    """
    Build human-readable feature names from a ColumnTransformer with num/cat pipelines.
    - 数值列：直接用原列名
    - 类别列：用 “列名=水平” 形式（RareCategoryGrouper 已合并稀有水平）
    """
    names = []
    for name, trans, cols in preprocess.transformers_:
        if name == "num":
            names.extend([str(c) for c in cols])
        elif name == "cat":
            oh = None
            if hasattr(trans, "named_steps"):
                oh = trans.named_steps.get("onehot")
            if oh is not None and hasattr(oh, "categories_"):
                for col, cats in zip(cols, oh.categories_):
                    names.extend([f"{col}={cat}" for cat in cats])
            else:
                names.extend([str(c) for c in cols])
        else:
            # passthrough/unknown
            names.extend([str(c) for c in cols])
    return names


def main(args) -> int:
    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap  # noqa: F401
    except Exception as e:
        raise RuntimeError("SHAP is not installed. Install `shap` in your env first.") from e

    import shap

    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {model_dir} (expected output from `ncdpipe fit`).")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cols = meta.get("columns_used") or []
    if not cols:
        raise ValueError("meta.json missing columns_used; re-run `ncdpipe fit`.")

    pipe = _load_pipeline(model_dir)
    preprocess = pipe.named_steps.get("preprocess")
    model = pipe.named_steps.get("model")

    if preprocess is None or model is None:
        raise ValueError("pipeline.joblib must be a Pipeline with steps: preprocess, model")

    # Load data for SHAP background/summary
    # Use the original (pre-preprocess) feature columns; the pipeline preprocessor handles encoding.
    from ncdpipe.io import apply_column_mapping, load_column_mapping

    mapping = load_column_mapping(args.mapping) if args.mapping else {}
    df = pd.read_excel(args.input_xlsx, sheet_name=args.sheet_name, na_values=args.missing_values)
    df = apply_column_mapping(df, mapping)

    # Ensure required columns exist
    X = df.copy()
    for c in cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[cols]

    # Sample for speed
    if args.sample and len(X) > args.sample:
        X = X.sample(n=args.sample, random_state=args.seed)

    X_trans = preprocess.transform(X)
    feature_names = _get_feature_names(preprocess)

    # TreeExplainer works for XGBoost models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # Handle binary classification output formats
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)

    # Importance table
    mean_abs = np.abs(shap_values).mean(axis=0)
    if feature_names and len(feature_names) == len(mean_abs):
        imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    else:
        imp = pd.DataFrame({"feature": [f"f{i}" for i in range(len(mean_abs))], "mean_abs_shap": mean_abs})

    # Map back到原始指标（数值列不变，类别列按“列名=”截断）
    def _source(name: str) -> str:
        return name.split("=", 1)[0] if "=" in name else name

    imp["source_feature"] = imp["feature"].apply(_source)
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(out_dir / "shap_importance.tsv", sep="\t", index=False, encoding="utf-8-sig")

    # Grouped importance by source feature（把同一原始指标的 one-hot/数值编码聚合）
    grouped = (
        imp.groupby("source_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    grouped.to_csv(out_dir / "shap_importance_grouped.tsv", sep="\t", index=False, encoding="utf-8-sig")

    # Cache for separate plotting
    cache_written = False
    if args.export_cache:
        np.save(out_dir / "shap_values.npy", shap_values)
        np.save(out_dir / "shap_X_trans.npy", X_trans)
        if feature_names:
            (out_dir / "shap_feature_names.json").write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
        cache_written = True

    # Plots (optional)
    if not args.no_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")

            # Force a CJK-capable font if available (avoid □ in plots)
            from matplotlib import font_manager

            def _force_font_by_path(path: str):
                p = Path(path)
                if not p.exists():
                    return None
                try:
                    font_manager.fontManager.addfont(str(p))
                except Exception:
                    pass
                try:
                    fp = font_manager.FontProperties(fname=str(p))
                    return fp.get_name()
                except Exception:
                    return None

            def _pick_cjk_font():
                candidates = {"Noto Sans Mono CJK SC", "Noto Sans CJK SC", "Noto Serif CJK SC"}
                for f in font_manager.fontManager.ttflist:
                    if f.name in candidates:
                        return f.name
                return None

            # Try explicit font file paths first (Ubuntu Noto CJK)
            font_name = None
            for path in [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            ]:
                font_name = _force_font_by_path(path)
                if font_name:
                    break

            if not font_name:
                # Fallback to any already-registered CJK font by name
                font_name = _pick_cjk_font()

            if font_name:
                matplotlib.rcParams["font.family"] = font_name
                matplotlib.rcParams["font.sans-serif"] = [font_name]
            else:
                matplotlib.rcParams["font.family"] = "DejaVu Sans"

            matplotlib.rcParams["axes.unicode_minus"] = False
            import matplotlib.pyplot as plt

            shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(out_dir / "shap_beeswarm.png", dpi=200)
            plt.close()

            shap.summary_plot(shap_values, X_trans, feature_names=feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(out_dir / "shap_bar.png", dpi=200)
            plt.close()
        except Exception as e:
            # Keep table output even if matplotlib is missing
            (out_dir / "plot_error.txt").write_text(str(e), encoding="utf-8")

    font_family = None
    if "matplotlib" in locals():
        try:
            font_family = matplotlib.rcParams.get("font.family")
        except Exception:
            font_family = None

    meta_out = {
        "model_dir": str(model_dir),
        "input_xlsx": str(Path(args.input_xlsx).resolve()),
        "sheet_name": args.sheet_name,
        "mapping": str(Path(args.mapping).resolve()) if args.mapping else None,
        "sample": int(args.sample) if args.sample else None,
        "seed": int(args.seed),
        "font_family": font_family,
        "cache_written": cache_written,
    }
    (out_dir / "shap_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {out_dir}/shap_importance.tsv")
    if (out_dir / "shap_beeswarm.png").exists():
        print(f"Wrote: {out_dir}/shap_beeswarm.png")
    if (out_dir / "shap_bar.png").exists():
        print(f"Wrote: {out_dir}/shap_bar.png")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute global SHAP (bar+beeswarm) for a fitted XGBoost pipeline.")
    parser.add_argument("--model_dir", required=True, help="Directory containing pipeline.joblib (from ncdpipe fit)")
    parser.add_argument("--input_xlsx", required=True, help="Input Excel used for SHAP background/summary")
    parser.add_argument("--sheet_name", default=0)
    parser.add_argument("--missing_values", nargs="*", default=["", "NA", "N/A", "null", "NULL", "999", "-1"])
    parser.add_argument("--mapping", default=None, help="Optional column mapping YAML/CSV (same format as specs/column_mapping.yaml)")
    parser.add_argument("--out_dir", required=True, help="Output directory for SHAP plots/tables")
    parser.add_argument("--sample", type=int, default=1000, help="Max rows to use for SHAP computation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export_cache", action="store_true", help="Save shap_values.npy, shap_X_trans.npy, shap_feature_names.json for separate plotting")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation (tables only)")
    raise SystemExit(main(parser.parse_args()))
