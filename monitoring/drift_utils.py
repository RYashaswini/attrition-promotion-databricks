import numpy as np
import pandas as pd

def calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI between a reference distribution and a current one, for one numeric feature."""
    ref = reference.dropna()
    cur = current.dropna()

    breakpoints = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(ref, breakpoints)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(ref))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(cur))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def flag_psi(psi: float) -> str:
    if psi < 0.1:
        return "STABLE"
    elif psi < 0.25:
        return "DRIFT_WARNING"
    return "DRIFT_ALERT"


def run_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(reference_df[col]):
            continue
        psi = calculate_psi(reference_df[col], current_df[col])
        rows.append({"feature": col, "psi": psi, "status": flag_psi(psi)})
    return pd.DataFrame(rows)