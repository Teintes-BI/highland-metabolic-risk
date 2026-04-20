from pathlib import Path
import pandas as pd


def read_table(path, file_type, encoding="utf-8", sheet_name=0, missing_values=None):
    if missing_values is None:
        missing_values = []

    if file_type == "csv":
        df = pd.read_csv(path, encoding=encoding, na_values=missing_values)
    elif file_type == "xlsx":
        df = pd.read_excel(path, sheet_name=sheet_name, na_values=missing_values)
    elif file_type == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return df


def load_column_mapping(path: str | Path | None) -> dict[str, str]:
    """
    Load a column rename mapping (from -> to).

    Supported formats:
    - YAML: either
        mappings: [{from: "...", to: "..."}, ...]
      or a plain dict {from: to}
    - CSV/TSV: columns must include `from` and `to`
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        obj = yaml.safe_load(p.read_text(encoding="utf-8"))
        if obj is None:
            return {}
        if isinstance(obj, dict) and "mappings" in obj:
            rows = obj.get("mappings") or []
            mapping = {}
            for r in rows:
                if not isinstance(r, dict) or "from" not in r or "to" not in r:
                    continue
                mapping[str(r["from"])] = str(r["to"])
            return mapping
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
        raise ValueError(f"Unsupported YAML mapping format: {type(obj)}")

    if p.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(p, sep=sep, dtype=str).fillna("")
        if not {"from", "to"}.issubset(df.columns):
            raise ValueError(f"{p} must have columns: from,to")
        mapping = {}
        for _, r in df.iterrows():
            if r["from"] and r["to"]:
                mapping[str(r["from"])] = str(r["to"])
        return mapping

    raise ValueError(f"Unsupported mapping file type: {p.suffix}")


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    Rename columns according to mapping (from -> to). Non-existing keys are ignored.
    If multiple columns map to the same target name, later renames win (pandas behavior).
    """
    if not mapping:
        return df
    existing = {k: v for k, v in mapping.items() if k in df.columns and k != v}
    if not existing:
        return df
    return df.rename(columns=existing)


def infer_types(df):
    types = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "date"
        else:
            uniq = s.dropna().astype(str).nunique()
            types[col] = "categorical" if uniq <= 20 else "text"
    return types
