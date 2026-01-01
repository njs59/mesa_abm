
# plots/score_against_experiment.py
"""
Score ABM sweep outputs against experimental summary statistics (S0, S1, S2).

- Loads experimental summary from data/experimental/<COND>_summary_stats.csv
  For invasive: INV_summary_stats.csv
  For proliferative: PRO_summary_stats.csv (if/when available)

- Finds all tags under results/<COND>/<SWEEP>/<tag>/run_*/summary_S012.csv
- For each tag, computes ABM mean curve across replicates at the experimental
  hour grid, then computes an error metric vs experimental data.

Metrics:
  rmse_equal   : unweighted RMSE across concatenated [S0, S1, S2] vectors
  rmse_weighted: weighted RMSE with weights {S0:1, S1:2, S2:1} (tuneable)

Outputs:
  results/<COND>/<SWEEP>/_scores/score_<metric>.csv  (sorted best → worst)
  results/<COND>/<SWEEP>/_scores/score_<metric>.json (top‑K tags plus quick stats)

Run examples:
  python plots/score_against_experiment.py --condition invasive --sweep speed_adhesion --metric rmse_weighted
  python plots/score_against_experiment.py --condition invasive --sweep proliferation --metric rmse_weighted
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_experimental(cond: str, base: str = "data/experimental") -> pd.DataFrame:
    """
    Load experimental summary stats for the given condition and attach hours.

    Assumes columns: S0, S1, S2 (no time column).
    Uses 30‑minute cadence (∆t = 0.5 h) and starts at timepoint 22 for INV.
    """
    base = Path(base)
    fname = {"invasive": "INV_summary_stats.csv",
             "proliferative": "PRO_summary_stats.csv"}[cond]
    p = base / fname
    if not p.exists():
        raise FileNotFoundError(f"Experimental file not found: {p}")
    df = pd.read_csv(p).reset_index(drop=True)

    # Map row index to timepoints (INV currently 22.., adjust here if needed)
    start_tp = 22
    df.insert(0, "timepoint", range(start_tp, start_tp + len(df)))
    df.insert(1, "hours", df["timepoint"] * 0.5)  # ∆t = 30 min
    return df[["hours", "S0", "S1", "S2"]]


def mean_curve_for_tag(tag_dir: Path) -> pd.DataFrame:
    """
    Aggregate mean S0/S1/S2 vs hours across all replicates under tag_dir.
    """
    runs = sorted(tag_dir.glob("run_*/summary_S012.csv"))
    if not runs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(fp) for fp in runs], ignore_index=True)
    m = df.groupby("hours")[["S0", "S1", "S2"]].mean().reset_index()
    return m


def rmse(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> float:
    """
    Root mean squared error; optional weighting vector w (same shape as a).
    """
    if w is None:
        return float(np.sqrt(np.mean((a - b) ** 2)))
    return float(np.sqrt(np.mean(w * (a - b) ** 2)))


def score_grid(condition: str,
               sweep: str,
               metric: str = "rmse_weighted",
               weights: dict[str, float] = {"S0": 1.0, "S1": 2.0, "S2": 1.0},
               results_root: str = "results",
               out_topk: int = 10) -> None:
    """
    Score all tags under results/<condition>/<sweep> against experimental data.

    Writes sorted CSV and JSON to results/<condition>/<sweep>/_scores/.
    Prints the top tags and scores at the end.
    """
    # Load experiment and set up targets
    exp = load_experimental(condition)
    hrs = exp["hours"].values
    exp_vec = np.concatenate([exp["S0"].values, exp["S1"].values, exp["S2"].values])

    # Build weights for weighted metric
    w = None
    if metric == "rmse_weighted":
        w = np.concatenate([
            np.full_like(exp["S0"].values, weights["S0"], dtype=float),
            np.full_like(exp["S1"].values, weights["S1"], dtype=float),
            np.full_like(exp["S2"].values, weights["S2"], dtype=float),
        ])

    base = Path(results_root) / condition / sweep
    rows = []

    # Iterate tags and compute score
    for tag_dir in sorted([d for d in base.iterdir() if d.is_dir() and d.name != "_scores"]):
        m = mean_curve_for_tag(tag_dir)
        if m.empty:
            continue

        # Interpolate ABM mean curve at the experimental hour grid
        m = m.set_index("hours").sort_index()
        abm_s0 = np.interp(hrs, m.index.values, m["S0"].values)
        abm_s1 = np.interp(hrs, m.index.values, m["S1"].values)
        abm_s2 = np.interp(hrs, m.index.values, m["S2"].values)
        abm_vec = np.concatenate([abm_s0, abm_s1, abm_s2])

        score = rmse(abm_vec, exp_vec, w=None if metric == "rmse_equal" else w)
        rows.append({"tag": tag_dir.name, metric: score})

    # Sort and write outputs
    score_df = pd.DataFrame(rows).sort_values(metric, ascending=True)
    out_dir = base / "_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(out_dir / f"score_{metric}.csv", index=False)

    top = score_df.head(out_topk).to_dict(orient="records")
    with open(out_dir / f"score_{metric}.json", "w") as f:
        json.dump({"condition": condition,
                   "sweep": sweep,
                   "metric": metric,
                   "weights": weights,
                   "top": top},
                  f, indent=2)

    # Informative prints (so you don't need to open files)
    print(f"[saved] {out_dir / f'score_{metric}.csv'}")
    print(f"[saved] {out_dir / f'score_{metric}.json'}")

    # ✅ Final print block showing the best tags right away
    print("\nTop tags by", metric)
    for rec in top:
        print(f"  {rec['tag']}: {rec[metric]:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score ABM sweeps vs experimental S0, S1, S2.")
    ap.add_argument("--condition", required=True, choices=["proliferative", "invasive"])
    ap.add_argument("--sweep", required=True, choices=["speed_adhesion", "proliferation", "fragmentation", "density"])
    ap.add_argument("--metric", choices=["rmse_equal", "rmse_weighted"], default="rmse_weighted")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    score_grid(args.condition, args.sweep, args.metric, results_root=args.results_root, out_topk=args.topk)


if __name__ == "__main__":
    main()
