from pathlib import Path
import pandas as pd
import numpy as np

from .utils import ensure_dir
from .io import apply_column_mapping, infer_types, load_column_mapping


def run_qc(specs, project_root: Path):
    from .io import read_table

    data = specs["data"]
    mapping_path = (specs.get("validation") or {}).get("external_mapping_path")
    mapping = load_column_mapping(mapping_path) if mapping_path else {}
    df = read_table(
        data["path"],
        data["type"],
        encoding=data.get("encoding", "utf-8"),
        sheet_name=data.get("sheet_name", 0),
        missing_values=data.get("missing_values", [])
    )
    df = apply_column_mapping(df, mapping)

    qc_dir = project_root / "outputs" / "qc"
    ensure_dir(qc_dir)

    shape_df = pd.DataFrame({
        "metric": ["rows", "columns"],
        "value": [df.shape[0], df.shape[1]]
    })
    shape_df.to_csv(qc_dir / "shape.tsv", sep="\t", index=False, encoding="utf-8-sig")

    types = infer_types(df)
    rows = []
    for col in df.columns:
        s = df[col]
        missing = s.isna().mean()
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "inferred_type": types[col],
            "missing_rate": round(float(missing), 6),
            "unique_count": int(s.nunique(dropna=True))
        })

    pd.DataFrame(rows).sort_values("missing_rate", ascending=False).to_csv(
        qc_dir / "columns.tsv", sep="\t", index=False, encoding="utf-8-sig"
    )

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    out_rows = []
    for col in num_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        q = s.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
        out_rows.append({
            "column": col,
            "min": float(s.min()),
            "p01": float(q.loc[0.01]),
            "p05": float(q.loc[0.05]),
            "p50": float(q.loc[0.5]),
            "p95": float(q.loc[0.95]),
            "p99": float(q.loc[0.99]),
            "max": float(s.max())
        })

    pd.DataFrame(out_rows).to_csv(qc_dir / "numeric_ranges.tsv", sep="\t", index=False, encoding="utf-8-sig")

    dup = df.duplicated().sum()
    pd.DataFrame([{"duplicate_rows": int(dup)}]).to_csv(
        qc_dir / "duplicates.tsv", sep="\t", index=False, encoding="utf-8-sig"
    )

    return qc_dir
