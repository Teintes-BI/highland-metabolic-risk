import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def binned_stats(df, bins=10):
    df = df.copy()
    df = df.sort_values("prob")
    df["bin"] = pd.qcut(df["prob"], q=bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True)
    out = grouped.apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "pos": g["y_true"].sum(),
                "pos_rate": g["y_true"].mean(),
                "prob_mean": g["prob"].mean(),
                "prob_median": g["prob"].median(),
            }
        )
    ).reset_index()
    return out


def compute_metrics(df):
    y = df["y_true"].values
    p = df["prob"].values
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return {
        "n": len(df),
        "pos": float(y.sum()),
        "pos_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def main(models_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    binned_rows = []

    for path in glob.glob(str(models_dir / "*/*/*/fold_predictions.tsv")):
        path = Path(path)
        parts = path.parts
        # .../models/<outcome>/<featureset>/<model>/fold_predictions.tsv
        outcome, featureset, model = parts[-4:-1]
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(subset=["prob", "y_true"])
        if df.empty:
            continue
        metrics = compute_metrics(df)
        metrics.update({"outcome": outcome, "featureset": featureset, "model": model})
        summary_rows.append(metrics)

        bins = binned_stats(df, bins=10)
        bins["outcome"] = outcome
        bins["featureset"] = featureset
        bins["model"] = model
        binned_rows.append(bins)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "calibration_summary.tsv", sep="\t", index=False, encoding="utf-8-sig")
    if binned_rows:
        pd.concat(binned_rows, ignore_index=True).to_csv(
            out_dir / "calibration_binned.tsv", sep="\t", index=False, encoding="utf-8-sig"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate calibration metrics from fold_predictions.tsv")
    parser.add_argument("--models_dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/calibration"))
    args = parser.parse_args()
    main(args.models_dir, args.out_dir)
