import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_label_map(path: Path | None) -> dict:
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
        for a, b in [("from", "to"), ("feature", "short"), ("feature", "label")]:
            if a in df.columns and b in df.columns:
                return {str(r[a]): str(r[b]) for _, r in df.iterrows()}
        raise ValueError("CSV/TSV must contain columns: from,to (or feature,short)")
    raise ValueError(f"Unsupported mapping file: {path}")


def _apply_label_map(feature_names: list[str], mapping: dict) -> list[str]:
    if not mapping:
        return feature_names
    out = []
    for name in feature_names:
        if "=" in name:
            base, level = name.split("=", 1)
            base_short = mapping.get(base, base)
            out.append(f"{base_short}={level}")
        else:
            out.append(mapping.get(name, name))
    return out


def _set_cjk_font():
    import matplotlib
    from matplotlib import font_manager

    matplotlib.use("Agg")

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

    font_name = None
    for path in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ]:
        font_name = _force_font_by_path(path)
        if font_name:
            break
    if not font_name:
        font_name = _pick_cjk_font()
    if font_name:
        matplotlib.rcParams["font.family"] = font_name
        matplotlib.rcParams["font.sans-serif"] = [font_name]
    else:
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return matplotlib


def main(args) -> int:
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir) if args.out_dir else cache_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    shap_values = np.load(cache_dir / "shap_values.npy")
    X_trans = np.load(cache_dir / "shap_X_trans.npy")
    feature_names = json.loads((cache_dir / "shap_feature_names.json").read_text(encoding="utf-8"))

    mapping = _load_label_map(Path(args.label_map)) if args.label_map else {}
    mapped_names = _apply_label_map(feature_names, mapping)

    # Recompute importance with mapped names
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({"feature": mapped_names, "mean_abs_shap": mean_abs})
    imp["source_feature"] = imp["feature"].apply(lambda x: x.split("=", 1)[0] if "=" in x else x)
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(out_dir / "shap_importance_mapped.tsv", sep="\t", index=False, encoding="utf-8-sig")
    grouped = (
        imp.groupby("source_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    grouped.to_csv(out_dir / "shap_importance_grouped_mapped.tsv", sep="\t", index=False, encoding="utf-8-sig")

    # Plot
    import shap

    _set_cjk_font()
    import matplotlib.pyplot as plt

    shap.summary_plot(shap_values, X_trans, feature_names=mapped_names, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm_mapped.png", dpi=200)
    plt.close()

    shap.summary_plot(shap_values, X_trans, feature_names=mapped_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bar_mapped.png", dpi=200)
    plt.close()

    print(f"Wrote: {out_dir}/shap_beeswarm_mapped.png")
    print(f"Wrote: {out_dir}/shap_bar_mapped.png")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot SHAP from cached arrays, with optional label mapping.")
    parser.add_argument("--cache_dir", required=True, help="Folder containing shap_values.npy, shap_X_trans.npy, shap_feature_names.json")
    parser.add_argument("--out_dir", default=None, help="Output directory for plots/tables (default: cache_dir)")
    parser.add_argument("--label_map", default=None, help="YAML/CSV/TSV mapping from original feature to short label")
    raise SystemExit(main(parser.parse_args()))
