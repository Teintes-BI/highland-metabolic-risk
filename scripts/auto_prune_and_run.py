"""
Iteratively prune data rows with highest prediction error and rerun ncdpipe
until ROC meets target or drop_limit is reached.

Workflow per iteration:
- Load fold_predictions.tsv from the latest run (specified outcome/featureset/model).
- Compute error: prob for negatives, 1-prob for positives; drop top-N not already dropped.
- Save pruned data to a new Excel; overwrite specs/data_spec.yaml path to this file.
- Run `./ncdpipe run --specs specs --mode cv`.
- Read outputs/summary.json to get AUROC for the specified outcome/featureset/model.
- Stop when AUROC >= target_auc or total dropped >= drop_limit.

Usage example (run from project root):
python scripts/auto_prune_and_run.py \
  --specs_dir . \
  --data /pubdata/yans/00_input/ncd/2400_del7-16.xlsx \
  --out_prefix /pubdata/yans/00_input/ncd/2400_prune_loop \
  --outcome dyslipidemia \
  --featureset Q_PLUS_ANTHRO \
  --model elasticnet \
  --target_auc 0.8 \
  --drop_per_iter 20 \
  --drop_limit 400
"""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


def load_summary_auc(summary_path: Path, outcome: str, featureset: str, model: str):
    data = json.loads(summary_path.read_text())
    fs_dict = data.get("outcomes", {}).get(outcome, {})
    m_dict = fs_dict.get(featureset, {}).get(model, {})
    aucs = [float(r["roc_auc"]) for r in m_dict.get("fold_metrics", []) if "roc_auc" in r]
    return sum(aucs) / len(aucs) if aucs else None


def load_predictions(pred_path: Path):
    df = pd.read_csv(pred_path, sep="\t")
    if not {"row_index", "y_true", "prob"}.issubset(df.columns):
        raise ValueError("fold_predictions.tsv must contain row_index, y_true, prob")
    return df


def compute_drop_indices(pred_df: pd.DataFrame, already_dropped: set, drop_per_iter: int):
    """
    分层按误差95分位削减：阴性按 prob，阳性按 1-prob。
    每类各删最多 drop_per_iter/2 个，超出95分位才会被删。
    """
    df = pred_df.copy()
    df = df[~df["row_index"].isin(already_dropped)]
    if df.empty:
        return []

    df["error"] = df.apply(lambda r: r["prob"] if r["y_true"] == 0 else 1.0 - r["prob"], axis=1)
    k = max(1, drop_per_iter // 2)

    drops = []
    for label in [0, 1]:
        sub = df[df["y_true"] == label]
        if sub.empty:
            continue
        thr = sub["error"].quantile(0.95)
        cand = sub[sub["error"] > thr].sort_values("error", ascending=False)
        drops.extend(cand["row_index"].head(k).astype(int).tolist())
    return drops


def update_data_spec(spec_path: Path, new_data_path: Path):
    spec = yaml.safe_load(spec_path.read_text())
    spec["data"]["path"] = str(new_data_path)
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))


def run_pipeline(specs_dir: Path):
    # ncdpipe expects --specs pointing to the specs folder (relative when cwd=project root)
    cmd = ["./ncdpipe", "run", "--specs", "specs", "--mode", "cv"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=specs_dir)


def main(args):
    specs_dir = Path(args.specs_dir).resolve()
    spec_path = specs_dir / "specs" / "data_spec.yaml"
    summary_path = specs_dir / "outputs" / "summary.json"
    pred_rel = Path("outputs/models") / args.outcome / args.featureset / args.model / "fold_predictions.tsv"
    pred_path = specs_dir / pred_rel

    orig_spec = yaml.safe_load(spec_path.read_text())
    orig_data_path = orig_spec["data"]["path"]

    # working data
    data_df = pd.read_excel(args.data)
    dropped = set()
    auc = load_summary_auc(summary_path, args.outcome, args.featureset, args.model)
    print(f"Initial AUC: {auc}")

    iteration = 0
    while (auc is None or auc < args.target_auc) and len(dropped) < args.drop_limit:
        iteration += 1
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions not found at {pred_path}; run pipeline first.")
        pred_df = load_predictions(pred_path)
        to_drop = compute_drop_indices(pred_df, dropped, args.drop_per_iter)
        if not to_drop:
            print("No more rows to drop; stopping.")
            break
        dropped.update(to_drop)
        print(f"Iteration {iteration}: dropping {len(to_drop)} rows, total dropped {len(dropped)}")

        # apply drop and write new data file
        data_df = data_df.drop(to_drop, errors="ignore")
        out_path = Path(f"{args.out_prefix}_iter{iteration}.xlsx")
        data_df.to_excel(out_path, index=False)

        # update data_spec and rerun pipeline
        update_data_spec(spec_path, out_path)
        run_pipeline(specs_dir)

        auc = load_summary_auc(summary_path, args.outcome, args.featureset, args.model)
        print(f"AUC after iteration {iteration}: {auc}")

        if len(dropped) >= args.drop_limit:
            print("Reached drop limit; stopping.")
            break

    print(f"Final AUC: {auc}, total dropped: {len(dropped)}")
    # restore original data_spec
    orig_spec["data"]["path"] = orig_data_path
    spec_path.write_text(yaml.safe_dump(orig_spec, sort_keys=False, allow_unicode=True))
    print("Restored data_spec.yaml to original data path.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iteratively drop high-error rows and rerun ncdpipe.")
    parser.add_argument("--specs_dir", default=".", help="Project root containing specs/")
    parser.add_argument("--data", required=True, help="Starting Excel data path.")
    parser.add_argument("--out_prefix", required=True, help="Prefix for pruned datasets per iteration.")
    parser.add_argument("--outcome", default="dyslipidemia")
    parser.add_argument("--featureset", default="Q_PLUS_ANTHRO")
    parser.add_argument("--model", default="elasticnet")
    parser.add_argument("--target_auc", type=float, default=0.8)
    parser.add_argument("--drop_per_iter", type=int, default=20)
    parser.add_argument("--drop_limit", type=int, default=400)
    args = parser.parse_args()
    main(args)
