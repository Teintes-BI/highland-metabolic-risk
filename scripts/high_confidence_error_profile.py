import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def read_data_from_specs(project_root: Path) -> pd.DataFrame:
    spec_path = project_root / "specs" / "data_spec.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    data = spec["data"]
    if data.get("type") != "xlsx":
        raise ValueError(f"Only xlsx is supported here; got: {data.get('type')}")
    return pd.read_excel(data["path"], sheet_name=data.get("sheet_name", 0), na_values=data.get("missing_values", []))


def safe_select_rows(df: pd.DataFrame, row_index: list[int]) -> pd.DataFrame:
    try:
        return df.loc[row_index].copy()
    except KeyError:
        # fallback: treat as positional indices
        idx = [i for i in row_index if 0 <= i < len(df)]
        return df.iloc[idx].copy()


def summarize_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "column": col,
                "n": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "median": float(s.median()) if s.notna().any() else np.nan,
                "p10": float(s.quantile(0.10)) if s.notna().any() else np.nan,
                "p90": float(s.quantile(0.90)) if s.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=["column", "value", "n", "pct"])
    s = df[col].astype("object")
    vc = s.value_counts(dropna=False)
    out = vc.reset_index()
    out.columns = ["value", "n"]
    out["pct"] = out["n"] / len(df)
    out.insert(0, "column", col)
    return out


def main(args) -> int:
    project_root = Path(args.project_root).resolve()
    out_dir = project_root / "outputs" / "analysis" / "high_confidence_errors"
    out_dir.mkdir(parents=True, exist_ok=True)

    wide_path = project_root / args.wide
    if not wide_path.exists():
        raise FileNotFoundError(f"Missing wide predictions: {wide_path}. Run multi_outcome_qc.py first.")
    wide = pd.read_csv(wide_path, sep="\t")

    data = read_data_from_specs(project_root)

    # columns to inspect (keep consistent with your feature sets + key clinical context)
    key_cols = [
        "城市/农村",
        "年龄",
        "6、A4 您的性别：",
        "8、A5 民族：",
        "11、A8 您本人文化程度：",
        "15、A12 就业状况：",
        "16、A13 去年您全家一年的人均可支配/纯收入(包括各种来源)在什么范围?",
        "17、B1  您最近30天的吸烟情况：",
        "21、B3 您过去是否饮过酒？",
        "22、B3.1 您的饮酒习惯属于下列哪一类？",
        "25、B4 过去一个月，您大概多少时间吃一次新鲜蔬菜？",
        "26、B5 过去一个月，您大概多少时间吃一次新鲜水果？",
        "27、B6 过去一个月，您大概多少时间吃一次糌(Zān)粑？",
        "28、(1)B7 过去一个月，您喝奶茶大概吃了___饼酥油？",
        "体质指数BMI（kg/m²）",
        "腰围（cm）",
        # optional context columns (may be leakage-blocked in modeling, but useful for subgroup interpretation)
        "SBP 收缩压",
        "DBP舒张压",
        "葡萄糖-血糖mmol/L",
        "总胆固醇mmol/L(TC)",
        "甘油三脂mmol/L(TG)",
        "血清低密度脂蛋白胆固醇mmol/L(LDL)",
        "血清高密度脂蛋白胆固醇mmol/L(HDL)",
    ]

    outcomes = args.outcomes or ["hypertension", "hyperglycemia", "dyslipidemia"]
    prob_variant = (args.prob_variant or "raw").strip().lower()
    for outcome in outcomes:
        y_col = f"{outcome}_y_true"
        p_col = f"{outcome}_prob" if prob_variant == "raw" else f"{outcome}_prob_calibrated"
        if y_col not in wide.columns or p_col not in wide.columns:
            continue

        y = wide[y_col].astype(float)
        p = wide[p_col].astype(float)
        row_index = wide["row_index"].astype(int)

        hc_fn = wide[(y == 1.0) & (p <= args.fn_prob)]
        hc_fp = wide[(y == 0.0) & (p >= args.fp_prob)]

        # write row lists
        for tag, sub in [("HC_FN", hc_fn), ("HC_FP", hc_fp)]:
            if sub.empty:
                continue
            idx = sub["row_index"].astype(int).tolist()
            sub_data = safe_select_rows(data, idx)
            keep = [c for c in key_cols if c in sub_data.columns]
            out = sub[["row_index", y_col, p_col]].copy()
            out = out.rename(columns={y_col: "y_true", p_col: "prob"})
            out = out.merge(sub_data[keep], left_on="row_index", right_index=True, how="left")
            out.to_csv(out_dir / f"{outcome}_{tag}_rows.tsv", sep="\t", index=False, encoding="utf-8-sig")

        # summary: compare HC_FN vs other positives vs negatives
        groups = {
            "HC_FN": hc_fn,
            "OTHER_POS": wide[(y == 1.0) & ~((y == 1.0) & (p <= args.fn_prob))],
            "NEG": wide[y == 0.0],
        }
        numeric_cols = [c for c in key_cols if c in data.columns]
        numeric_cols = [c for c in numeric_cols if c not in {"城市/农村", "6、A4 您的性别：", "8、A5 民族："}]

        summary_blocks = []
        for gname, gwide in groups.items():
            if gwide.empty:
                continue
            idx = gwide["row_index"].astype(int).tolist()
            gdata = safe_select_rows(data, idx)
            num = summarize_numeric(gdata, numeric_cols)
            num.insert(0, "group", gname)
            summary_blocks.append(num)

        if summary_blocks:
            pd.concat(summary_blocks, ignore_index=True).to_csv(
                out_dir / f"{outcome}_numeric_summary.tsv", sep="\t", index=False, encoding="utf-8-sig"
            )

        # categorical distributions (diet + smoking + drinking)
        cat_cols = [
            "城市/农村",
            "6、A4 您的性别：",
            "8、A5 民族：",
            "17、B1  您最近30天的吸烟情况：",
            "21、B3 您过去是否饮过酒？",
            "22、B3.1 您的饮酒习惯属于下列哪一类？",
            "25、B4 过去一个月，您大概多少时间吃一次新鲜蔬菜？",
            "26、B5 过去一个月，您大概多少时间吃一次新鲜水果？",
            "27、B6 过去一个月，您大概多少时间吃一次糌(Zān)粑？",
            "28、(1)B7 过去一个月，您喝奶茶大概吃了___饼酥油？",
        ]
        cat_out_dir = out_dir / f"{outcome}_categorical"
        cat_out_dir.mkdir(parents=True, exist_ok=True)
        for gname, gwide in groups.items():
            if gwide.empty:
                continue
            idx = gwide["row_index"].astype(int).tolist()
            gdata = safe_select_rows(data, idx)
            cat_tables = []
            for col in cat_cols:
                tab = summarize_categorical(gdata, col)
                if tab.empty:
                    continue
                tab.insert(0, "group", gname)
                cat_tables.append(tab)
            if cat_tables:
                pd.concat(cat_tables, ignore_index=True).to_csv(
                    cat_out_dir / f"{gname}.tsv", sep="\t", index=False, encoding="utf-8-sig"
                )

    print(f"Wrote high-confidence error profiles to: {out_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile high-confidence FP/FN rows per outcome using wide CV predictions.")
    parser.add_argument("--project_root", default=".", help="Project root containing specs/ and outputs/")
    parser.add_argument(
        "--wide",
        default="outputs/qc/selected_fold_predictions_wide.tsv",
        help="Wide predictions table from scripts/multi_outcome_qc.py",
    )
    parser.add_argument("--outcomes", nargs="*", default=None, help="Subset outcomes to analyze")
    parser.add_argument(
        "--prob_variant",
        choices=["raw", "calibrated"],
        default="raw",
        help="Use raw probs (<outcome>_prob) or calibrated probs (<outcome>_prob_calibrated) from the wide table.",
    )
    parser.add_argument("--fp_prob", type=float, default=0.9, help="High-confidence FP if y=0 and prob>=this")
    parser.add_argument("--fn_prob", type=float, default=0.1, help="High-confidence FN if y=1 and prob<=this")
    raise SystemExit(main(parser.parse_args()))
