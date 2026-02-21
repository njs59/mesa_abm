#!/usr/bin/env python3
"""
Collect and organise MLE outputs across scenarios and models.
Applies tolerance thresholding to ODE parameters, builds a consolidated
results DataFrame, and tracks global counters.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import json
import os

import numpy as np
import pandas as pd


ZERO_TOL = 1e-10  # values below this (in abs) for ODE params are considered 0


@dataclass
class MLEAggregator:
    run_dir: str
    swept_keys: List[str]
    rows: List[Dict[str, Any]] = field(default_factory=list)
    total_abm_runs: int = 0
    total_mle_fits: int = 0

    def add_abm_run(self):
        self.total_abm_runs += 1

    def add_record(
        self,
        *,
        scenario_name: str,
        mode: str,
        model_key: str,
        overrides: Dict[str, Any],
        mle_out: Dict[str, Any],
        ode_param_names: List[str],
    ):
        """Add one MLE result row (applies ZERO_TOL to ODE params only)."""
        self.total_mle_fits += 1

        theta = np.asarray(mle_out["theta_hat"], dtype=float)
        k_model = len(ode_param_names)
        ode_vals = theta[:k_model].copy()
        # threshold small values for ODE params only
        ode_vals[np.abs(ode_vals) < ZERO_TOL] = 0.0
        # noise terms kept as-is
        sigmas = theta[k_model:]

        row: Dict[str, Any] = {
            "scenario": scenario_name,
            "mode": mode,
            "model": model_key,
            "AIC": float(mle_out.get("AIC", float("nan"))),
            "BIC": float(mle_out.get("BIC", float("nan"))),
            "max_loglik": float(mle_out.get("max_loglik", float("nan"))),
        }
        # ABM override values (swept keys)
        for k in self.swept_keys:
            row[f"abm::{k}"] = overrides.get(k, None)
        # ODE parameters (thresholded)
        for name, val in zip(ode_param_names, ode_vals):
            row[f"ode::{name}"] = float(val)
        # Sigma noise terms (kept for completeness)
        sigma_names = ["sigma0", "sigma1", "sigma2"]
        for i, sval in enumerate(sigmas):
            if i < len(sigma_names):
                row[f"noise::{sigma_names[i]}"] = float(sval)
            else:
                row[f"noise::sigma{i}"] = float(sval)
        self.rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame()
        return pd.DataFrame(self.rows)

    def save_results_csv(self, path: str) -> str:
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return path

    def save_counters(self):
        payload = {
            "total_abm_runs": int(self.total_abm_runs),
            "total_mle_fits": int(self.total_mle_fits),
        }
        with open(os.path.join(self.run_dir, "counters.json"), "w") as f:
            json.dump(payload, f, indent=2)
        return payload


__all__ = ["MLEAggregator", "ZERO_TOL"]