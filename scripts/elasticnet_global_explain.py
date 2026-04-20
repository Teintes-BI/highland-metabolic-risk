import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure local ncdpipe package is importable when loading pipeline.joblib
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_pipeline(model_dir: Path):
    import joblib

    return joblib.load(model_dir / "pipeline.joblib")


def _get_feature_names_from_preprocess(preprocess) -> list[str] | None:
    """
    Build readable feature names from ColumnTransformer(num,cat) pipelines.
    - numeric: original column name
    - categorical: "col=level" from OneHotEncoder categories_
    """
    try:
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
                names.extend([str(c) for c in cols])
        return names
    except Exception:
        return None


def _load_label_map(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        if isinstance(data, list):
            out = {}
            for row in data:
                if isinstance(row, dict) and "from" in row and "to" in row:
                    out[str(row["from"])] = str(row["to"])
            return out
        return {}
    if path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
        if {"from", "to"}.issubset(df.columns):
            return {str(r["from"]): str(r["to"]) for _, r in df.iterrows()}
        if {"feature", "short"}.issubset(df.columns):
            return {str(r["feature"]): str(r["short"]) for _, r in df.iterrows()}
        raise ValueError("CSV/TSV must contain columns: from,to (or feature,short)")
    raise ValueError(f"Unsupported mapping file: {path}")


def main(args) -> int:
    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else model_dir / "interpret_elasticnet"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {model_dir} (expected output from `ncdpipe fit`).")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    pipe = _load_pipeline(model_dir)
    preprocess = pipe.named_steps.get("preprocess")
    model = pipe.named_steps.get("model")
    if preprocess is None or model is None:
        raise ValueError("pipeline.joblib must be a Pipeline with steps: preprocess, model")

    if not hasattr(model, "coef_"):
        raise ValueError("This does not look like an ElasticNet LogisticRegression model (missing coef_)")

    feature_names = _get_feature_names_from_preprocess(preprocess)
    coef = np.asarray(model.coef_).reshape(-1)

    if not feature_names or len(feature_names) != len(coef):
        feature_names = [f"f{i}" for i in range(len(coef))]

    df = pd.DataFrame({"feature": feature_names, "coef": coef})
    df["abs_coef"] = df["coef"].abs()
    df["odds_ratio"] = np.exp(df["coef"])
    df["direction"] = np.where(df["coef"] >= 0, "risk↑", "risk↓")
    df["source_feature"] = df["feature"].apply(lambda x: x.split("=", 1)[0] if "=" in x else x)

    label_map = _load_label_map(Path(args.label_map)) if args.label_map else {}
    if label_map:
        def _label(name: str) -> str:
            if "=" in name:
                base, level = name.split("=", 1)
                return f"{label_map.get(base, base)}={level}"
            return label_map.get(name, name)

        df["feature_label"] = df["feature"].apply(_label)
        df["source_label"] = df["source_feature"].apply(lambda x: label_map.get(x, x))
    else:
        df["feature_label"] = df["feature"]
        df["source_label"] = df["source_feature"]

    df = df.sort_values("abs_coef", ascending=False)
    df.to_csv(out_dir / "elasticnet_coef.tsv", sep="\t", index=False, encoding="utf-8-sig")

    # Grouped by source feature: sum of abs(coef) across one-hot levels
    grouped = (
        df.groupby(["source_feature", "source_label"], as_index=False)["abs_coef"]
        .sum()
        .rename(columns={"abs_coef": "sum_abs_coef"})
        .sort_values("sum_abs_coef", ascending=False)
    )
    grouped.to_csv(out_dir / "elasticnet_coef_grouped.tsv", sep="\t", index=False, encoding="utf-8-sig")

    # Optional bar plot for grouped importance (top-k)
    if not args.no_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            topk = grouped.head(int(args.top))
            plt.figure(figsize=(10, max(4, 0.25 * len(topk))))
            plt.barh(topk["source_label"][::-1], topk["sum_abs_coef"][::-1])
            plt.xlabel("Sum |coef| (grouped)")
            plt.tight_layout()
            plt.savefig(out_dir / "elasticnet_coef_grouped_bar.png", dpi=200)
            plt.close()
        except Exception as e:
            (out_dir / "plot_error.txt").write_text(str(e), encoding="utf-8")

    meta_out = {
        "model_dir": str(model_dir),
        "out_dir": str(out_dir),
        "n_features": int(len(coef)),
        "top": int(args.top),
        "label_map": str(Path(args.label_map).resolve()) if args.label_map else None,
        "meta": {
            "tag": meta.get("tag"),
            "outcome": meta.get("outcome"),
            "featureset": meta.get("featureset"),
            "model": meta.get("model"),
            "n_train": meta.get("n_train"),
            "pos_rate": meta.get("pos_rate"),
        },
    }
    (out_dir / "elasticnet_explain_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {out_dir}/elasticnet_coef.tsv")
    print(f"Wrote: {out_dir}/elasticnet_coef_grouped.tsv")
    if (out_dir / "elasticnet_coef_grouped_bar.png").exists():
        print(f"Wrote: {out_dir}/elasticnet_coef_grouped_bar.png")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global interpretability for fitted ElasticNet (coef-based).")
    parser.add_argument("--model_dir", required=True, help="Directory containing pipeline.joblib (from `ncdpipe fit`) for elasticnet")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: <model_dir>/interpret_elasticnet)")
    parser.add_argument("--label_map", default=None, help="YAML/CSV/TSV mapping from original feature to short label")
    parser.add_argument("--top", type=int, default=30, help="Top-k for grouped bar plot")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation (tables only)")
    raise SystemExit(main(parser.parse_args()))

