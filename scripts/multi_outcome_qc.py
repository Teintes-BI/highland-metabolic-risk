import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelRef:
    outcome: str
    featureset: str
    model: str

    @property
    def rel_pred_path(self) -> Path:
        return Path("outputs/models") / self.outcome / self.featureset / self.model / "fold_predictions.tsv"


def _mean_metric(fold_metrics: list[dict], key: str) -> float | None:
    vals = [m.get(key) for m in (fold_metrics or [])]
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else None


def load_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def list_model_refs(summary: dict) -> list[ModelRef]:
    out = []
    for outcome, fs_map in (summary.get("outcomes") or {}).items():
        for featureset, m_map in (fs_map or {}).items():
            for model in (m_map or {}).keys():
                out.append(ModelRef(outcome=outcome, featureset=featureset, model=model))
    return out


def choose_best_by_auc(summary: dict, restrict_featuresets: set[str] | None = None) -> list[ModelRef]:
    chosen = []
    for outcome, fs_map in (summary.get("outcomes") or {}).items():
        best = None
        best_auc = -1.0
        for featureset, m_map in (fs_map or {}).items():
            if restrict_featuresets is not None and featureset not in restrict_featuresets:
                continue
            for model, payload in (m_map or {}).items():
                mean_auc = _mean_metric(payload.get("fold_metrics", []), "roc_auc")
                if mean_auc is None:
                    continue
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best = ModelRef(outcome=outcome, featureset=featureset, model=model)
        if best is not None:
            chosen.append(best)
    return chosen


def parse_picks(picks: list[str]) -> list[ModelRef]:
    out = []
    for s in picks or []:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(f"--pick must be outcome:featureset:model, got: {s}")
        out.append(ModelRef(outcome=parts[0], featureset=parts[1], model=parts[2]))
    return out


def load_predictions(project_root: Path, ref: ModelRef) -> pd.DataFrame:
    p = project_root / ref.rel_pred_path
    if not p.exists():
        raise FileNotFoundError(f"Missing predictions for {ref}: {p}")
    df = pd.read_csv(p, sep="\t")
    need = {"fold", "row_index", "y_true", "prob"}
    if not need.issubset(df.columns):
        raise ValueError(f"{p} must contain columns {sorted(need)}; got {sorted(df.columns)}")
    keep = ["fold", "row_index", "y_true", "prob"]
    if "prob_calibrated" in df.columns:
        keep.append("prob_calibrated")
    df = df[keep].copy()
    df["outcome"] = ref.outcome
    df["featureset"] = ref.featureset
    df["model"] = ref.model
    df["y_true"] = df["y_true"].astype(float)
    df["prob"] = df["prob"].astype(float)
    if "prob_calibrated" in df.columns:
        df["prob_calibrated"] = pd.to_numeric(df["prob_calibrated"], errors="coerce")
    return df


def main(args) -> int:
    project_root = Path(args.project_root).resolve()
    summary_path = project_root / args.summary
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)

    picks = parse_picks(args.pick)
    if picks:
        selected = picks
    else:
        restrict = set(args.restrict_featureset) if args.restrict_featureset else None
        selected = choose_best_by_auc(summary, restrict_featuresets=restrict)
        if not selected:
            raise ValueError(f"No runnable outcomes found in {summary_path}")

    # best models table (uses summary fold_metrics if available)
    rows = []
    for ref in selected:
        payload = (
            (summary.get("outcomes") or {})
            .get(ref.outcome, {})
            .get(ref.featureset, {})
            .get(ref.model, {})
        )
        fold_metrics = payload.get("fold_metrics", []) if isinstance(payload, dict) else []
        rows.append(
            {
                "outcome": ref.outcome,
                "featureset": ref.featureset,
                "model": ref.model,
                "mean_roc_auc": _mean_metric(fold_metrics, "roc_auc"),
                "mean_pr_auc": _mean_metric(fold_metrics, "pr_auc"),
                "n_folds": len([m for m in fold_metrics if m.get("roc_auc") is not None]),
                "pred_path": str(ref.rel_pred_path),
            }
        )
    best_df = pd.DataFrame(rows).sort_values(["outcome", "mean_roc_auc"], ascending=[True, False])
    best_df.to_csv(out_dir / "best_models.tsv", sep="\t", index=False, encoding="utf-8-sig")

    # load fold predictions for selected refs
    long_preds = pd.concat([load_predictions(project_root, r) for r in selected], ignore_index=True)
    prob_variant = (args.prob_variant or "raw").strip().lower()
    if prob_variant == "calibrated":
        if "prob_calibrated" not in long_preds.columns:
            raise ValueError("prob_calibrated not found in fold_predictions.tsv; rerun 07 pipeline with recalibration enabled.")
        long_preds["prob_used"] = long_preds["prob_calibrated"]
    else:
        long_preds["prob_used"] = long_preds["prob"]
    long_preds["error"] = np.where(long_preds["y_true"] == 1.0, 1.0 - long_preds["prob_used"], long_preds["prob_used"])
    long_preds.to_csv(out_dir / "selected_fold_predictions_long.tsv", sep="\t", index=False, encoding="utf-8-sig")

    # misclassification summary per outcome
    fp_thr = float(args.fp_prob)
    fn_thr = float(args.fn_prob)
    err_thr = float(args.error_thr)

    summary_rows = []
    mis_rows = []
    for outcome, g in long_preds.groupby("outcome", observed=True):
        y = g["y_true"].values
        p = g["prob_used"].values
        fp = int(((y == 0.0) & (p >= fp_thr)).sum())
        fn = int(((y == 1.0) & (p <= fn_thr)).sum())
        bad = g[g["error"] >= err_thr].copy()
        if not bad.empty:
            bad["bad_type"] = np.where(bad["y_true"] == 1.0, "FN_like", "FP_like")
            mis_rows.append(bad)
        summary_rows.append(
            {
                "outcome": outcome,
                "n": int(len(g)),
                "pos": int((y == 1.0).sum()),
                "pos_rate": float((y == 1.0).mean()) if len(y) else np.nan,
                "prob_variant": prob_variant,
                f"fp_prob>={fp_thr:g}": fp,
                f"fn_prob<={fn_thr:g}": fn,
                f"error>={err_thr:g}": int(len(bad)),
            }
        )
    pd.DataFrame(summary_rows).sort_values("outcome").to_csv(
        out_dir / "misclassification_summary.tsv", sep="\t", index=False, encoding="utf-8-sig"
    )

    if mis_rows:
        pd.concat(mis_rows, ignore_index=True).sort_values(["outcome", "error"], ascending=[True, False]).to_csv(
            out_dir / "high_error_rows.tsv", sep="\t", index=False, encoding="utf-8-sig"
        )

    # optional wide table (only if >=2 outcomes)
    outcomes = sorted(long_preds["outcome"].unique().tolist())
    if len(outcomes) >= 2:
        wide = None
        for outcome in outcomes:
            cols = ["row_index", "fold", "y_true", "prob"]
            if "prob_calibrated" in long_preds.columns:
                cols.append("prob_calibrated")
            g = long_preds[long_preds["outcome"] == outcome][cols].copy()
            # each row_index should appear exactly once per outcome in CV predictions
            dup = g["row_index"].duplicated().sum()
            if dup:
                agg = {"fold": "first", "y_true": "first", "prob": "mean"}
                if "prob_calibrated" in g.columns:
                    agg["prob_calibrated"] = "mean"
                g = g.groupby("row_index", as_index=False).agg(agg)
            g = g.rename(
                columns={
                    "fold": f"{outcome}_fold",
                    "y_true": f"{outcome}_y_true",
                    "prob": f"{outcome}_prob",
                    "prob_calibrated": f"{outcome}_prob_calibrated",
                }
            )
            wide = g if wide is None else wide.merge(g, on=["row_index"], how="outer")
        wide["prob_variant"] = prob_variant
        wide.to_csv(out_dir / "selected_fold_predictions_wide.tsv", sep="\t", index=False, encoding="utf-8-sig")

    print(f"Wrote: {out_dir}/best_models.tsv")
    print(f"Wrote: {out_dir}/misclassification_summary.tsv")
    print(f"Wrote: {out_dir}/selected_fold_predictions_long.tsv")
    if (out_dir / "high_error_rows.tsv").exists():
        print(f"Wrote: {out_dir}/high_error_rows.tsv")
    if (out_dir / "selected_fold_predictions_wide.tsv").exists():
        print(f"Wrote: {out_dir}/selected_fold_predictions_wide.tsv")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-outcome QC from outputs/summary.json + fold_predictions.tsv")
    parser.add_argument("--project_root", default=".", help="Project root containing outputs/ and specs/")
    parser.add_argument("--summary", default="outputs/summary.json", help="Path to summary.json (relative to project_root)")
    parser.add_argument("--out_dir", default="outputs/qc", help="Output folder (relative to project_root)")
    parser.add_argument(
        "--pick",
        action="append",
        default=[],
        help="Explicit model pick: outcome:featureset:model (repeatable). If omitted, auto-picks best AUROC per outcome.",
    )
    parser.add_argument(
        "--restrict_featureset",
        action="append",
        default=[],
        help="Restrict auto-pick to these feature set names (repeatable), e.g. --restrict_featureset Q_ONLY.",
    )
    parser.add_argument(
        "--prob_variant",
        choices=["raw", "calibrated"],
        default="raw",
        help="Which probability to use for QC thresholds: raw uses 'prob'; calibrated uses 'prob_calibrated' (if present).",
    )
    parser.add_argument("--fp_prob", type=float, default=0.9, help="Flag FPs when y=0 and prob>=this")
    parser.add_argument("--fn_prob", type=float, default=0.1, help="Flag FNs when y=1 and prob<=this")
    parser.add_argument("--error_thr", type=float, default=0.9, help="Flag high-error rows when error>=this (error=prob if y=0 else 1-prob)")
    raise SystemExit(main(parser.parse_args()))
