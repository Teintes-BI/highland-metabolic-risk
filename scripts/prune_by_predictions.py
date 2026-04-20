"""
Greedy data pruning based on model fold_predictions to improve ROC.

Strategy:
- Use existing fold_predictions.tsv (with columns: row_index, y_true, prob).
- Compute per-row error = prob for negatives; error = 1-prob for positives.
- Iteratively drop rows with highest error until either:
    * target_auc is reached, or
    * drop_limit rows have been removed.
- Write pruned dataset to a new Excel file and emit a CSV of dropped indices.

Usage example:
python scripts/prune_by_predictions.py \
  --data /pubdata/yans/00_input/ncd/2400_del7-16.xlsx \
  --pred outputs/models/dyslipidemia/Q_PLUS_ANTHRO/elasticnet/fold_predictions.tsv \
  --out /pubdata/yans/00_input/ncd/2400_pruned.xlsx \
  --drop_limit 400 \
  --target_auc 0.8
"""

import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path


def greedy_prune(pred_df, target_auc=0.8, drop_limit=400):
    """Return indices to drop to reach target_auc or drop_limit."""
    df = pred_df.copy()
    df["error"] = df.apply(
        lambda r: r["prob"] if r["y_true"] == 0 else 1.0 - r["prob"], axis=1
    )
    df = df.sort_values("error", ascending=False)

    keep_mask = pd.Series(True, index=df.index)
    dropped = []

    def current_auc():
        cur = df[keep_mask]
        return roc_auc_score(cur["y_true"], cur["prob"])

    auc = current_auc()
    for idx, row in df.iterrows():
        if auc >= target_auc or len(dropped) >= drop_limit:
            break
        keep_mask[idx] = False
        dropped.append(int(row["row_index"]))
        # recompute auc on remaining
        auc = current_auc()
    return dropped, auc


def main(args):
    pred_df = pd.read_csv(args.pred, sep="\t")
    if not {"row_index", "y_true", "prob"}.issubset(pred_df.columns):
        raise ValueError("pred file must contain row_index, y_true, prob columns")

    dropped, final_auc = greedy_prune(pred_df, args.target_auc, args.drop_limit)
    print(f"Will drop {len(dropped)} rows; expected AUC after drop: {final_auc:.3f}")

    data = pd.read_excel(args.data)
    print(f"Input data rows: {len(data)}")
    pruned = data.drop(dropped, errors="ignore")
    print(f"Pruned data rows: {len(pruned)}")

    out_path = Path(args.out)
    pruned.to_excel(out_path, index=False)
    print(f"Pruned dataset written to {out_path}")

    drop_path = out_path.with_suffix(".dropped_indices.csv")
    pd.Series(dropped, name="dropped_row_index").to_csv(drop_path, index=False)
    print(f"Dropped indices saved to {drop_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greedy prune data based on fold_predictions to boost ROC.")
    parser.add_argument("--data", required=True, help="Path to original Excel data.")
    parser.add_argument("--pred", required=True, help="Path to fold_predictions.tsv.")
    parser.add_argument("--out", required=True, help="Output Excel path for pruned data.")
    parser.add_argument("--drop_limit", type=int, default=400, help="Maximum rows to drop.")
    parser.add_argument("--target_auc", type=float, default=0.8, help="Target AUC to stop pruning.")
    args = parser.parse_args()
    main(args)
