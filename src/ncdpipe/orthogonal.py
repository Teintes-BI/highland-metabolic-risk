import numpy as np
import pandas as pd


def bin_variable(series, rule):
    if rule["type"] == "numeric":
        bins = rule["bins"]
        labels = rule.get("labels")
        edges = [-np.inf] + bins
        labels = labels if labels is not None else list(range(1, len(edges)))
        # ensure labels count matches bins-1
        if len(labels) != len(edges) - 1:
            # truncate or pad labels
            if len(labels) > len(edges) - 1:
                labels = labels[: len(edges) - 1]
            else:
                labels = labels + [f"bin{i}" for i in range(len(labels) + 1, len(edges))]
        return pd.cut(series, bins=edges, labels=labels, include_lowest=True)
    return series.astype(str)


def orthogonal_tables(df, outcome_col, combos, binning, min_cell_n=50, model=None):
    outputs = {}
    for combo in combos:
        name = combo["name"]
        variables = combo["variables"]
        tmp = df.copy()
        for v in variables:
            if v in binning:
                tmp[v] = bin_variable(tmp[v], binning[v])
            else:
                tmp[v] = tmp[v].astype(str)

        group = tmp.groupby(variables, dropna=False)
        table = group[outcome_col].agg(["count", "mean"]).reset_index()
        table.rename(columns={"count": "n", "mean": "rate"}, inplace=True)
        table["sparse"] = table["n"] < min_cell_n

        if model is not None:
            try:
                proba = model.predict_proba(df.drop(columns=[outcome_col]))[:, 1]
                tmp["pred"] = proba
                adj = tmp.groupby(variables, dropna=False)["pred"].mean().reset_index()
                table = table.merge(adj, on=variables, how="left")
            except Exception:
                table["pred"] = np.nan

        outputs[name] = table
    return outputs
