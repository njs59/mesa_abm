
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare NND with and without radius jitter within the integer-size rounding band,
using parameters bootstrapped (with replacement, by weight) from a pyabc final population.

Outputs:
  - <out_dir>/nnd_values.csv    (all NNDs; long/tidy)
  - <out_dir>/nnd_summary.csv   (per-replicate summaries + Δ medians)
  - <out_dir>/nnd_hist_overlay.png
  - <out_dir>/nnd_medians.png

Run:
    python scripts/compare_radius_jitter.py \
        --db results/your_abc.db \
        --pkg abm \
        --n-boot 1000 \
        --out-dir results_nnd \
        --units "µm"

If your ABM package is under 'src/abm', add:
    --pkg-path src

Assumptions:
  - Your ABM package (e.g., 'abm') has:
        clusters_model.py defining ClustersModel
        utils.py defining DEFAULTS and radius_from_size_3d
  - ClustersModel.__init__ uses Mesa 3 AgentSet API and initialises:
        from mesa.agent import AgentSet
        self.agents = AgentSet(self)
    before spawning agents.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sqlite3
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Ensure the project root (parent of 'scripts') is on sys.path automatically
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------- Utility helpers ----------------------------- #

def deep_copy_jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    """Lossless deep copy for JSON-like dicts."""
    return json.loads(json.dumps(d))


def _safe_cast_num(x: Any) -> Any:
    """Best-effort numeric cast (handles strings of ints/floats)."""
    if isinstance(x, (int, float)) or x is None:
        return x
    if isinstance(x, str):
        try:
            if x.strip().isdigit():
                return int(x)
            return float(x)
        except Exception:
            return x
    return x


def apply_dotted_params(base: Dict[str, Any], theta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply dotted keys like 'merge.p_merge' into nested dict 'base'.
    Performs a deep copy first; attempts numeric cast of values.
    """
    params = deep_copy_jsonable(base)
    for k, v in (theta or {}).items():
        if not isinstance(k, str):
            continue
        path = k.split(".")
        ref = params
        for key in path[:-1]:
            if key not in ref or not isinstance(ref[key], dict):
                ref[key] = {}
            ref = ref[key]
        ref[path[-1]] = _safe_cast_num(v)
    return params


def extract_steps(theta: Dict[str, Any], default_steps: int) -> int:
    """
    Infer number of timesteps from common keys used in ABC; fallback to default.
    """
    for key in ["time.steps", "steps", "n_steps", "T", "time_steps"]:
        if key in (theta or {}):
            try:
                return int(float(theta[key]))
            except Exception:
                pass
    return int(default_steps)


def _list_tables_and_cols(cur) -> Tuple[List[str], Dict[str, List[str]]]:
    """Return all table names and a dict of table->column names."""
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    cols = {}
    for t in tables:
        cols[t] = [r[1] for r in cur.execute(f"PRAGMA table_info({t});")]
    return tables, cols


def _find_table(tables: List[str], candidates: List[str]) -> str | None:
    """Find a table by exact lower-case name, else by fuzzy contains."""
    lower = {t.lower(): t for t in tables}
    for c in candidates:
        if c in lower:
            return lower[c]
    # fuzzy fallback
    for t in tables:
        for c in candidates:
            if c in t.lower():
                return t
    return None


def _first_present(candidates: List[str], cols: List[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _resolve_db_path(db_arg: str) -> str:
    """
    Accept either a filesystem path or a SQLAlchemy-style sqlite URI.
    Returns an absolute filesystem path and checks existence with a helpful error.
    """
    db_str = os.path.expandvars(os.path.expanduser(db_arg))
    if db_str.startswith("sqlite:///"):
        db_path = db_str[len("sqlite:///"):]
    elif db_str.startswith("sqlite://"):
        db_path = db_str[len("sqlite://"):]
    else:
        db_path = db_str
    db_path = os.path.abspath(db_path)
    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"Database file not found: {db_path}\n"
            f"Tip: pass a real file path (or a sqlite:/// URI). "
            f"Current working dir: {os.getcwd()}"
        )
    return db_path


# ---------------------- Robust pyabc final-pop reader ---------------------- #

def read_final_population_pyabc(db_path: str) -> pd.DataFrame:
    """
    Robust reader for pyabc-like SQLite DBs.
    Auto-detects population / model / particle / parameter table names and layouts.
    Returns DataFrame with columns: weight, params (dict).

    Handles schemas where:
      - 'populations' has 't' (iteration) and 'id';
      - 'particles' has neither 't' nor population FK, but references 'models';
      - 'models' references 'populations' via 'population_id';
      - parameters are either JSON-in-particle or in a normalised 'parameters' table.
    """
    db_abs = _resolve_db_path(db_path)
    con = sqlite3.connect(db_abs)
    cur = con.cursor()

    tables, table_cols = _list_tables_and_cols(cur)

    # 1) Locate population table & determine final population (by t or by id)
    pop_tbl = _find_table(tables, ["population", "populations", "abc_population", "pop"])
    if pop_tbl is None:
        con.close()
        raise RuntimeError(
            "No population-like table found. "
            f"Tables present: {tables}. Expected one of: population / populations / abc_population."
        )
    pop_cols = table_cols[pop_tbl]
    t_col = _first_present(["t", "iteration", "gen", "generation"], pop_cols)
    pop_id_col = _first_present(["id", "population_id", "pop_id"], pop_cols)

    if t_col is not None:
        tmax = cur.execute(f"SELECT MAX({t_col}) FROM {pop_tbl}").fetchone()[0]
        if tmax is None:
            con.close()
            raise RuntimeError(f"No rows in {pop_tbl}.")
        # Map to population id if present
        if pop_id_col is not None:
            row = cur.execute(
                f"SELECT {pop_id_col} FROM {pop_tbl} WHERE {t_col} = ?", (tmax,)
            ).fetchone()
            pop_id_val = row[0] if row else None
        else:
            pop_id_val = None
    else:
        if pop_id_col is None:
            con.close()
            raise RuntimeError(
                f"Table {pop_tbl} has no iteration column and no obvious id column. "
                f"Columns: {pop_cols}"
            )
        pop_id_val = cur.execute(
            f"SELECT MAX({pop_id_col}) FROM {pop_tbl}"
        ).fetchone()[0]
        if pop_id_val is None:
            con.close()
            raise RuntimeError(f"No rows in {pop_tbl}.")
        tmax = None  # will filter particles by population id instead

    # 2) Locate particle table
    particle_tbl = _find_table(tables, ["particle", "particles", "abc_particle"])
    if particle_tbl is None:
        con.close()
        raise RuntimeError(
            f"No particle-like table found. Tables present: {tables}. "
            "Expected one of: particle / particles / abc_particle."
        )
    particle_cols = table_cols[particle_tbl]
    weight_col = _first_present(["weight", "w", "w_norm"], particle_cols)
    if weight_col is None:
        con.close()
        raise RuntimeError(
            f"Could not find a weight column on '{particle_tbl}'. "
            f"Columns: {particle_cols}. Expected one of: weight / w / w_norm."
        )

    # Try filtering particles by t on particle, or by a population FK...
    t_on_particle = _first_present(["t", "iteration", "gen", "generation"], particle_cols)
    particle_fk_to_pop = _first_present(["population", "population_id", "pop_id", "pop"], particle_cols)

    particle_df = None
    if t_on_particle is not None and tmax is not None:
        rows = cur.execute(
            f"SELECT * FROM {particle_tbl} WHERE {t_on_particle} = ?", (tmax,)
        ).fetchall()
        particle_df = pd.DataFrame(rows, columns=particle_cols)
    elif particle_fk_to_pop is not None and pop_id_val is not None:
        rows = cur.execute(
            f"SELECT * FROM {particle_tbl} WHERE {particle_fk_to_pop} = ?", (pop_id_val,)
        ).fetchall()
        particle_df = pd.DataFrame(rows, columns=particle_cols)

    # ...otherwise, try the models → particles path (your schema)
    if particle_df is None or particle_df.empty:
        models_tbl = _find_table(tables, ["models", "model", "abc_models"])
        if models_tbl is None:
            con.close()
            raise RuntimeError(
                "Could not determine how to select the final particle set, and no 'models' table found.\n"
                f"- population table columns: {pop_cols}\n"
                f"- particle table columns:   {particle_cols}\n"
                "Need either 't' on particle, a particle FK to population, or a models(population_id) join."
            )
        models_cols = table_cols[models_tbl]
        model_id_col = _first_present(["id", "model_id"], models_cols)
        model_pop_fk = _first_present(["population_id", "pop_id", "population"], models_cols)
        particle_model_fk = _first_present(["model_id", "model"], particle_cols)
        if not (model_id_col and model_pop_fk and particle_model_fk):
            con.close()
            raise RuntimeError(
                f"'models' table or foreign keys not recognised.\n"
                f"models columns:   {models_cols}\n"
                f"particles columns:{particle_cols}\n"
                "Expected models(population_id) and particles(model_id)."
            )
        if pop_id_val is None and t_col is not None:
            # Map tmax to pop_id_val if needed
            row = cur.execute(
                f"SELECT {pop_id_col} FROM {pop_tbl} WHERE {t_col} = ?", (tmax,)
            ).fetchone()
            pop_id_val = row[0] if row else None
        # Fetch model ids for final population
        model_rows = cur.execute(
            f"SELECT {model_id_col} FROM {models_tbl} WHERE {model_pop_fk} = ?", (pop_id_val,)
        ).fetchall()
        model_ids = [r[0] for r in model_rows]
        if not model_ids:
            con.close()
            raise RuntimeError(f"No models found for population id {pop_id_val}.")
        q_marks = ",".join(["?"] * len(model_ids))
        rows = cur.execute(
            f"SELECT * FROM {particle_tbl} WHERE {particle_model_fk} IN ({q_marks})",
            model_ids
        ).fetchall()
        particle_df = pd.DataFrame(rows, columns=particle_cols)

    if particle_df.empty:
        con.close()
        raise RuntimeError("Final particle set is empty after filtering.")

    # 3) Parameters: JSON-in-particle or normalised table
    json_param_col = _first_present(["parameter", "parameters_json"], particle_cols)
    if json_param_col:
        recs = []
        for _, r in particle_df.iterrows():
            w = float(r[weight_col])
            pjson = r[json_param_col]
            try:
                params = json.loads(pjson) if isinstance(pjson, (str, bytes)) else {}
            except Exception:
                params = {}
            recs.append({"weight": w, "params": params})
        df = pd.DataFrame(recs)
        con.close()
        wv = np.asarray(df["weight"], dtype=float)
        df["weight"] = wv / wv.sum() if wv.sum() > 0 else wv
        return df

    # normalised parameter table
    param_tbl = _find_table(tables, ["parameter", "parameters", "abc_parameter", "param"])
    if param_tbl is None:
        con.close()
        raise RuntimeError(
            "No parameter table found (normalised layout). "
            f"Tables present: {tables}. Expected 'parameter' / 'parameters'."
        )
    param_cols = table_cols[param_tbl]
    particle_id_col = _first_present(["particle_id", "id", "pid"], particle_cols)
    if particle_id_col is None:
        con.close()
        raise RuntimeError(
            f"Could not find particle id column on '{particle_tbl}'. "
            f"Columns: {particle_cols}. Expected one of: particle_id / id / pid."
        )
    param_fk = _first_present(["particle", "particle_id", "pid"], param_cols)
    name_col = _first_present(["name", "key"], param_cols)
    value_col = _first_present(["value", "val"], param_cols)
    if not (param_fk and name_col and value_col):
        con.close()
        raise RuntimeError(
            f"Parameter table '{param_tbl}' must have FK to particle and name/value columns. "
            f"Columns: {param_cols}"
        )

    pids = particle_df[particle_id_col].unique().tolist()
    if not pids:
        con.close()
        raise RuntimeError("No particle ids present for final population.")
    q_marks = ",".join(["?"] * len(pids))
    rows_param = cur.execute(
        f"SELECT {param_fk}, {name_col}, {value_col} FROM {param_tbl} "
        f"WHERE {param_fk} IN ({q_marks})", pids
    ).fetchall()
    con.close()

    df_param = pd.DataFrame(rows_param, columns=["pid", "name", "value"])
    if df_param.empty:
        # It is possible no parameters were stored; proceed with empty dicts
        df_w = particle_df[[particle_id_col, weight_col]].rename(
            columns={particle_id_col: "pid", weight_col: "weight"}
        )
        recs = [{"weight": float(w), "params": {}} for w in df_w["weight"]]
        df = pd.DataFrame(recs)
        wv = np.asarray(df["weight"], dtype=float)
        df["weight"] = wv / wv.sum() if wv.sum() > 0 else wv
        return df

    df_piv = df_param.pivot_table(index="pid", columns="name",
                                  values="value", aggfunc="last").reset_index()
    df_w = particle_df[[particle_id_col, weight_col]].rename(
        columns={particle_id_col: "pid", weight_col: "weight"}
    )
    df_m = df_w.merge(df_piv, on="pid", how="left")

    recs = []
    for _, r in df_m.iterrows():
        # Pack all param columns (except pid/weight); safe-cast strings to numbers
        p = {k: _safe_cast_num(r[k]) for k in df_m.columns if k not in ("pid", "weight") and pd.notnull(r[k])}
        recs.append({"weight": float(r["weight"]), "params": p})

    df = pd.DataFrame(recs)
    wv = np.asarray(df["weight"], dtype=float)
    df["weight"] = wv / wv.sum() if wv.sum() > 0 else wv
    return df


# --------------------------- Radius jitter helpers ------------------------- #

def radius_bounds_for_size(n: int, cell_volume: float) -> Tuple[float, float]:
    """
    [r_min, r_max) such that any radius in this interval rounds to integer size n,
    assuming size = round( (4/3)π r^3 / cell_volume ).
    """
    n = max(int(n), 0)
    vmin = max((n - 0.5) * cell_volume, 0.0)
    vmax = (n + 0.5) * cell_volume
    c = 3.0 / (4.0 * math.pi)
    rmin = (c * vmin) ** (1.0 / 3.0) if vmin > 0 else 0.0
    rmax = (c * vmax) ** (1.0 / 3.0)
    return float(rmin), float(rmax)


def varied_radius(n: int, cell_volume: float, rng: np.random.Generator) -> float:
    rmin, rmax = radius_bounds_for_size(n, cell_volume)
    span = max(rmax - rmin, 0.0)
    u = rng.random()
    return float(rmin + u * span)


# ------------------------------- Geometry/NND ------------------------------- #

def torus_nnd(pos: np.ndarray, width: float, height: float) -> np.ndarray:
    """
    Toroidal nearest-neighbour distance for points in pos (N,2).
    Vectorised O(N^2), fine for N up to a few thousand.
    """
    if pos is None or len(pos) <= 1:
        return np.array([], dtype=float)
    pos = np.asarray(pos, dtype=float)
    dx = np.abs(pos[:, None, 0] - pos[None, :, 0])
    dy = np.abs(pos[:, None, 1] - pos[None, :, 1])
    dx = np.minimum(dx, width - dx)
    dy = np.minimum(dy, height - dy)
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1)


# ---------------------------------- Main ----------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Forward-simulate NND from pyabc final population (varied vs non-varied radii)."
    )
    ap.add_argument("--db", required=True, help="Path to pyabc .db file (or sqlite:/// URI)")
    ap.add_argument("--pkg", required=True, help="ABM package name (e.g., 'abm')")
    ap.add_argument("--pkg-path", default=None,
                    help="Directory that CONTAINS the ABM package (e.g., '.', or 'src').")
    ap.add_argument("--n-boot", type=int, default=1000, help="Number of bootstrap replicates (default 1000)")
    ap.add_argument("--out-dir", default="results_nnd", help="Output directory for CSVs and plots")
    ap.add_argument("--seed", type=int, default=12345, help="Master RNG seed for bootstrap sampling")
    ap.add_argument("--units", default="units", help="Axis label units for NND plots (e.g., 'µm')")
    args = ap.parse_args()

    # If user supplied a pkg-path (e.g., 'src'), prepend it
    if args.pkg_path:
        pkg_path = os.path.abspath(args.pkg_path)
        if pkg_path not in sys.path:
            sys.path.insert(0, pkg_path)

    os.makedirs(args.out_dir, exist_ok=True)

    # Import ABM modules dynamically
    try:
        mod_model = importlib.import_module(f"{args.pkg}.clusters_model")
        mod_utils = importlib.import_module(f"{args.pkg}.utils")
    except ModuleNotFoundError as e:
        print(f"[ERROR] Could not import package '{args.pkg}': {e}")
        print("Hint: ensure 'abm/__init__.py' exists and run from the project root, "
              "or pass --pkg-path to the directory that CONTAINS the package (e.g., '.' or 'src').")
        sys.exit(1)

    # Pull required symbols
    ClustersModel = getattr(mod_model, "ClustersModel", None)
    DEFAULTS = getattr(mod_utils, "DEFAULTS", None)
    radius_from_size_3d = getattr(mod_utils, "radius_from_size_3d", None)
    if ClustersModel is None or DEFAULTS is None or radius_from_size_3d is None:
        print("[ERROR] ABM package missing required symbols "
              "(ClustersModel, DEFAULTS, radius_from_size_3d).")
        sys.exit(1)

    # Resolve DB path early to fail fast if not present
    try:
        resolved_db = _resolve_db_path(args.db)
        print(f"[info] Using DB: {resolved_db}")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Read final population (weights + parameter dicts)
    try:
        final_pop = read_final_population_pyabc(resolved_db)
    except Exception as e:
        # Helpful dump of tables if schema is unexpected
        try:
            con = sqlite3.connect(resolved_db)
            cur = con.cursor()
            tabs = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            print(f"[diag] Tables present: {tabs}")
            for t in tabs:
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t});")]
                print(f"[diag] {t} columns: {cols}")
            con.close()
        except Exception:
            pass
        raise

    weights = final_pop["weight"].to_numpy()
    thetas = final_pop["params"].tolist()
    base_defaults = DEFAULTS

    rng_boot = np.random.default_rng(args.seed)

    all_nnd_records: List[Dict[str, Any]] = []
    all_sum_records: List[Dict[str, Any]] = []

    for b in range(int(args.n_boot)):
        # Weighted resample a particle
        idx = rng_boot.choice(len(thetas), p=weights)
        theta = thetas[idx] if isinstance(thetas[idx], dict) else {}

        # Apply dotted parameters to DEFAULTS
        params_theta = apply_dotted_params(base_defaults, theta)
        # Steps used during ABC (fallback to DEFAULTS)
        steps = extract_steps(theta, params_theta.get("time", {}).get("steps", 300))
        steps = int(steps)

        # Geometry & constants
        W = float(params_theta["space"]["width"])
        H = float(params_theta["space"]["height"])
        cv = float(params_theta["physics"]["cell_volume"])

        # Different seeds for the two paired runs
        seed_nonvar = 10_000 + b
        seed_var = 20_000 + b
        np_rng = np.random.default_rng(seed_var)

        # --- Non-varied radii model ---
        m0 = ClustersModel(params=params_theta, seed=seed_nonvar)
        for a in list(m0.agents):
            n = int(a.size)
            a.radius = float(radius_from_size_3d(n, cell_volume=cv))
        for _ in range(steps):
            m0.step()
        pos0 = m0.pos_log[-1] if len(m0.pos_log) else np.empty((0, 2))
        nnd0 = torus_nnd(pos0, W, H)

        # --- Varied radii model ---
        m1 = ClustersModel(params=params_theta, seed=seed_var)
        for a in list(m1.agents):
            n = int(a.size)
            a.radius = varied_radius(n, cv, np_rng)
        for _ in range(steps):
            m1.step()
        pos1 = m1.pos_log[-1] if len(m1.pos_log) else np.empty((0, 2))
        nnd1 = torus_nnd(pos1, W, H)

        # Summaries
        def summary(x: np.ndarray) -> Dict[str, float]:
            if x.size == 0:
                return {"median": np.nan, "mean": np.nan, "iqr": np.nan}
            q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
            return {"median": float(med), "mean": float(np.mean(x)), "iqr": float(q3 - q1)}

        s0, s1 = summary(nnd0), summary(nnd1)
        all_sum_records.append({
            "replicate": b,
            "steps": steps,
            "median_nonvar": s0["median"],
            "median_var": s1["median"],
            "mean_nonvar": s0["mean"],
            "mean_var": s1["mean"],
            "iqr_nonvar": s0["iqr"],
            "iqr_var": s1["iqr"],
            "delta_median": (s1["median"] - s0["median"]) if not math.isnan(s0["median"]) else np.nan
        })

        # Raw values
        all_nnd_records.extend({"replicate": b, "condition": "non_varied", "step": steps, "nnd": float(v)} for v in nnd0)
        all_nnd_records.extend({"replicate": b, "condition": "varied", "step": steps, "nnd": float(v)} for v in nnd1)

        if (b + 1) % 50 == 0 or b == 0:
            print(f"[{b+1}/{args.n_boot}] completed...")

    # Save CSVs
    df_vals = pd.DataFrame(all_nnd_records)
    df_sum = pd.DataFrame(all_sum_records)
    vals_csv = os.path.join(args.out_dir, "nnd_values.csv")
    sum_csv = os.path.join(args.out_dir, "nnd_summary.csv")
    df_vals.to_csv(vals_csv, index=False)
    df_sum.to_csv(sum_csv, index=False)
    print(f"Saved: {vals_csv}")
    print(f"Saved: {sum_csv}")

    # Plots (merged histograms; British English labels)
    def plot_histograms(df_vals: pd.DataFrame, out_png: str, units: str):
        plt.figure(figsize=(7, 5))
        if df_vals["nnd"].notna().sum() == 0:
            print("[WARN] No NND values to plot.")
            return
        mn, mx = df_vals["nnd"].min(), df_vals["nnd"].max()
        bins = np.linspace(mn, mx, 40) if np.isfinite(mn) and np.isfinite(mx) else 40
        for cond, colour, label in [("non_varied", "#1f77b4", "Non‑varied"),
                                    ("varied", "#d62728", "Varied")]:
            vals = df_vals.loc[df_vals["condition"] == cond, "nnd"].to_numpy()
            plt.hist(vals, bins=bins, alpha=0.5, label=label, color=colour,
                     edgecolor="k", linewidth=0.4)
        plt.xlabel(f"Nearest‑neighbour distance ({units})")
        plt.ylabel("Count (merged)")
        plt.title("Effect of radius jitter (within integer‑size rounding band) on NND")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    def plot_replicate_medians(df_sum: pd.DataFrame, out_png: str, units: str):
        plt.figure(figsize=(6, 5))
        x = np.array([0, 1])
        for _, r in df_sum.iterrows():
            if not (np.isnan(r["median_nonvar"]) or np.isnan(r["median_var"])):
                plt.plot(x, [r["median_nonvar"], r["median_var"]],
                         color="grey", alpha=0.2, linewidth=0.7)
        plt.scatter(np.zeros(len(df_sum)), df_sum["median_nonvar"], color="#1f77b4", s=10, label="Non‑varied")
        plt.scatter(np.ones(len(df_sum)), df_sum["median_var"], color="#d62728", s=10, label="Varied")
        plt.xticks([0, 1], ["Non‑varied", "Varied"])
        plt.ylabel(f"Replicate median NND ({units})")
        plt.title("Per‑replicate median NND (paired)")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    plot_histograms(df_vals, os.path.join(args.out_dir, "nnd_hist_overlay.png"), args.units)
    plot_replicate_medians(df_sum, os.path.join(args.out_dir, "nnd_medians.png"), args.units)
    print(f"Saved plots in: {args.out_dir}")

    # Console summary
    valid = df_sum["delta_median"].dropna()
    if len(valid):
        print(f"Mean Δ median NND (Varied − Non‑varied): {valid.mean():.3f} {args.units} (over {len(valid)} replicates)")
    else:
        print("No valid Δ median statistics available.")


if __name__ == "__main__":
    main()
