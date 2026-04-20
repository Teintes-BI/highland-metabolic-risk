import argparse
from pathlib import Path

import pandas as pd


def _prob_col(outcome: str, variant: str) -> str:
    if variant == "calibrated":
        return f"{outcome}_prob_calibrated"
    return f"{outcome}_prob"


def assign_tier(df: pd.DataFrame, prob_variant: str, htn_hi: float, hyper_hi: float, lip_hi: float, any_mid: float) -> pd.Series:
    h = df[_prob_col("hypertension", prob_variant)].astype(float)
    g = df[_prob_col("hyperglycemia", prob_variant)].astype(float)
    l = df[_prob_col("dyslipidemia", prob_variant)].astype(float)

    tier3 = (h >= htn_hi) & (g >= hyper_hi)
    tier2 = (~tier3) & ((h >= htn_hi) | (g >= hyper_hi) | (l >= lip_hi))
    tier1 = (~tier3) & (~tier2) & ((h >= any_mid) | (g >= any_mid) | (l >= any_mid))
    tier0 = ~(tier3 | tier2 | tier1)

    out = pd.Series("T0", index=df.index, dtype="object")
    out[tier1] = "T1"
    out[tier2] = "T2"
    out[tier3] = "T3"
    out[tier0] = "T0"
    return out


def tier_summary(df: pd.DataFrame, prob_variant: str) -> pd.DataFrame:
    out = []
    for tier, g in df.groupby("tier", observed=True):
        row = {"tier": tier, "n": int(len(g))}
        for outcome in ["hypertension", "hyperglycemia", "dyslipidemia"]:
            y = g[f"{outcome}_y_true"].astype(float)
            row[f"{outcome}_pos"] = int((y == 1.0).sum())
            row[f"{outcome}_pos_rate"] = float((y == 1.0).mean()) if len(y) else 0.0
            row[f"{outcome}_prob_mean"] = float(g[_prob_col(outcome, prob_variant)].astype(float).mean())
        out.append(row)
    return pd.DataFrame(out).sort_values("tier", ascending=False)


def main(args) -> int:
    project_root = Path(args.project_root).resolve()
    wide_path = project_root / args.wide
    if not wide_path.exists():
        raise FileNotFoundError(f"Missing wide predictions: {wide_path}. Run multi_outcome_qc.py first.")

    df = pd.read_csv(wide_path, sep="\t")

    prob_variant = (args.prob_variant or "raw").strip().lower()
    need = {
        "row_index",
        _prob_col("hypertension", prob_variant),
        _prob_col("hyperglycemia", prob_variant),
        _prob_col("dyslipidemia", prob_variant),
        "hypertension_y_true",
        "hyperglycemia_y_true",
        "dyslipidemia_y_true",
    }
    if not need.issubset(df.columns):
        raise ValueError(f"Wide file missing columns: {sorted(need - set(df.columns))}")

    df["tier"] = assign_tier(df, prob_variant, args.htn_hi, args.hyper_hi, args.lip_hi, args.any_mid)

    out_dir = project_root / "outputs" / "analysis" / "joint_risk"
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "row_index",
        "tier",
        _prob_col("hypertension", prob_variant),
        _prob_col("hyperglycemia", prob_variant),
        _prob_col("dyslipidemia", prob_variant),
        "hypertension_y_true",
        "hyperglycemia_y_true",
        "dyslipidemia_y_true",
    ]
    df[cols].assign(prob_variant=prob_variant).to_csv(out_dir / "joint_risk_ladder.tsv", sep="\t", index=False, encoding="utf-8-sig")

    summ = tier_summary(df, prob_variant).assign(prob_variant=prob_variant)
    summ.to_csv(out_dir / "tier_summary.tsv", sep="\t", index=False, encoding="utf-8-sig")

    # optionally emit highest-priority list
    if args.export_t3:
        df.loc[df["tier"] == "T3", ["row_index"]].to_csv(out_dir / "tier_T3_row_index.tsv", sep="\t", index=False, encoding="utf-8-sig")

    print(f"Wrote: {out_dir}/joint_risk_ladder.tsv")
    print(f"Wrote: {out_dir}/tier_summary.tsv")
    if args.export_t3:
        print(f"Wrote: {out_dir}/tier_T3_row_index.tsv")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a triage-oriented joint risk ladder from 3-outcome CV probabilities.")
    parser.add_argument("--project_root", default=".", help="Project root containing outputs/")
    parser.add_argument(
        "--wide",
        default="outputs/qc/selected_fold_predictions_wide.tsv",
        help="Wide predictions table from scripts/multi_outcome_qc.py",
    )
    parser.add_argument(
        "--prob_variant",
        choices=["raw", "calibrated"],
        default="raw",
        help="Use raw probs (<outcome>_prob) or calibrated probs (<outcome>_prob_calibrated) from the wide table.",
    )
    parser.add_argument("--htn_hi", type=float, default=0.8, help="High threshold for hypertension_prob")
    parser.add_argument("--hyper_hi", type=float, default=0.7, help="High threshold for hyperglycemia_prob")
    parser.add_argument("--lip_hi", type=float, default=0.7, help="High threshold for dyslipidemia_prob (lower priority)")
    parser.add_argument("--any_mid", type=float, default=0.6, help="Moderate threshold for any outcome prob")
    parser.add_argument("--export_t3", action="store_true", help="Export row_index list for Tier T3")
    raise SystemExit(main(parser.parse_args()))
