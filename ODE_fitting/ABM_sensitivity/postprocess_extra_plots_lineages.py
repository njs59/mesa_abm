#!/usr/bin/env python3
"""
Post-processing: lineage-aware pairwise coagulation AND size-based coagulation.

Outputs:
    <run>/extra_plots_lineages/<scenario>/pairwise_coagulation_vs_initial_distance.png
    <run>/extra_plots_lineages/<scenario>/coagulation_vs_size.png

Global overlays:
    <run>/extra_plots/pairwise_coagulation_vs_initial_distance_ALL_SCENARIOS.png
    <run>/extra_plots/coagulation_vs_size_ALL_SCENARIOS.png
"""

import os
import argparse
from typing import List, Dict, Tuple, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================================================
#  UNION–FIND
# ================================================================
class DSU:
    def __init__(self, elems: List[int]):
        self.parent = {int(e): int(e) for e in elems}
        self.rank = {int(e): 0 for e in elems}
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]: self.parent[ra] = rb
        elif self.rank[rb] < self.rank[ra]: self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

# ================================================================
#  HELPERS
# ================================================================
def read_timestep_csv(path: str):
    df = pd.read_csv(path)
    ids = df["id"].to_numpy(int)
    pos = df[["x","y"]].to_numpy(float)
    sizes = df["size"].to_numpy(float)
    return ids, pos, sizes

def nearest_index(x: np.ndarray, cands: np.ndarray) -> int:
    dif = cands - x[None,:]
    d2 = np.einsum("ij,ij->i", dif, dif)
    return int(np.argmin(d2))

# ================================================================
#  LINEAGE RECONSTRUCTION
# ================================================================
def reconstruct_lineages(rep_dir: str, size_tol=2.0):
    t_files = sorted(f for f in os.listdir(rep_dir)
                     if f.startswith("t_") and f.endswith(".csv"))
    if not t_files:
        raise RuntimeError(f"No timesteps found in {rep_dir}")

    # --- t0 ---
    ids0, pos0, sizes0 = read_timestep_csv(os.path.join(rep_dir, t_files[0]))
    initial_ids = ids0.tolist()
    initial_pos = {int(i): pos0[k].copy() for k,i in enumerate(ids0)}

    id_to_initials = {int(i): {int(i)} for i in ids0}
    dsu = DSU(initial_ids)

    prev_ids, prev_pos, prev_sizes = ids0, pos0, sizes0

    def idxmap(arr): return {int(v): int(i) for i,v in enumerate(arr)}

    for tf in t_files[1:]:
        next_ids, next_pos, next_sizes = read_timestep_csv(os.path.join(rep_dir, tf))

        prev_i = idxmap(prev_ids)
        next_i = idxmap(next_ids)

        set_prev = set(prev_ids.tolist())
        set_next = set(next_ids.tolist())

        persisting = set_prev & set_next
        lost = list(set_prev - set_next)
        new_ids = list(set_next - set_prev)

        # ---------------- FRAGMENTATION ----------------
        if new_ids:
            cand_ids, cand_pos = [], []
            for cid in persisting:
                ip = prev_i[cid]
                inx = next_i[cid]
                if next_sizes[inx] - prev_sizes[ip] <= (-1 + size_tol):
                    cand_ids.append(cid)
                    cand_pos.append(next_pos[inx])
            cand_ids = np.array(cand_ids, int)
            cand_pos = np.array(cand_pos, float) if len(cand_ids) else np.zeros((0,2))

            for nid in new_ids:
                inx = next_i[nid]
                p_new = next_pos[inx]
                if len(cand_ids)>0:
                    k = nearest_index(p_new, cand_pos)
                    parent = int(cand_ids[k])
                    id_to_initials[int(nid)] = set(id_to_initials[parent])
                else:
                    # fallback: nearest persisting
                    if persisting:
                        c2pos = np.array([next_pos[next_i[c]] for c in persisting])
                        kk = nearest_index(p_new, c2pos)
                        parent2 = list(persisting)[kk]
                        id_to_initials[int(nid)] = set(id_to_initials[parent2])
                    else:
                        id_to_initials[int(nid)] = set()

        # ---------------- MERGES ----------------
        rec_ids, rec_pos, rec_gain = [], [], []
        for cid in persisting:
            ip = prev_i[cid]
            inx = next_i[cid]
            gain = next_sizes[inx] - prev_sizes[ip]
            if gain >= (1 - size_tol):
                rec_ids.append(cid)
                rec_pos.append(next_pos[inx])
                rec_gain.append(float(gain))
        rec_ids = np.array(rec_ids, int)
        rec_pos = np.array(rec_pos, float) if len(rec_ids) else np.zeros((0,2))
        rec_gain = np.array(rec_gain, float)

        for lid in lost:
            p_lost = prev_pos[prev_i[lid]]
            if len(rec_ids)==0:
                if persisting:
                    c2 = np.array([next_pos[next_i[c]] for c in persisting])
                    kk = nearest_index(p_lost, c2)
                    recipient = int(list(persisting)[kk])
                else:
                    continue
            else:
                dif = rec_pos - p_lost[None,:]
                d2 = np.einsum("ij,ij->i", dif, dif)
                order = np.lexsort((-rec_gain, d2))
                recipient = int(rec_ids[order[0]])

            lost_init = id_to_initials.get(int(lid), set())
            rec_init = id_to_initials.get(int(recipient), set())

            if lost_init and rec_init:
                ia, ib = next(iter(lost_init)), next(iter(rec_init))
                dsu.union(ia, ib)
                id_to_initials[int(recipient)] = set(lost_init)|set(rec_init)

            id_to_initials.pop(int(lid), None)

        prev_ids, prev_pos, prev_sizes = next_ids, next_pos, next_sizes

    return dsu, initial_pos

# ================================================================
#  PAIRWISE MERGE STATS PER REPEAT
# ================================================================
def repeat_pair_stats(rep_dir: str, bins: np.ndarray, size_tol=2.0):
    dsu, pos0 = reconstruct_lineages(rep_dir, size_tol)
    init_ids = sorted(pos0.keys())
    P0 = np.array([pos0[i] for i in init_ids])

    merged, dists = [], []
    n = len(init_ids)

    for i in range(n):
        ia = init_ids[i]
        for j in range(i+1, n):
            ib = init_ids[j]
            dx = P0[i,0] - P0[j,0]
            dy = P0[i,1] - P0[j,1]
            d = np.hypot(dx, dy)
            dists.append(d)
            merged.append(1 if dsu.find(ia)==dsu.find(ib) else 0)

    dists = np.array(dists)
    merged = np.array(merged)

    idx = np.clip(np.searchsorted(bins, dists, side="right")-1, 0, len(bins)-2)
    total = np.bincount(idx, minlength=len(bins)-1)
    mergd = np.bincount(idx, weights=merged, minlength=len(bins)-1)

    return mergd.astype(int), total.astype(int)

# ================================================================
#  SIZE-BASED MERGE STATS PER REPEAT
# ================================================================
def repeat_size_stats(rep_dir: str):
    """
    Computes:
        For each cluster size s:
            merges[s] = number of merges involving size s
            exists[s] = number of timesteps where size s existed
    """
    t_files = sorted(f for f in os.listdir(rep_dir)
                     if f.startswith("t_") and f.endswith(".csv"))

    # size → count of merges
    merges_by_size = {}
    exists_by_size = {}

    for tf_prev, tf_next in zip(t_files[:-1], t_files[1:]):
        ids_prev, pos_prev, size_prev = read_timestep_csv(os.path.join(rep_dir, tf_prev))
        ids_next, pos_next, size_next = read_timestep_csv(os.path.join(rep_dir, tf_next))

        set_prev = set(ids_prev.tolist())
        set_next = set(ids_next.tolist())
        lost = list(set_prev - set_next)

        # count existence
        for s in size_prev:
            s_int = int(round(s))
            exists_by_size[s_int] = exists_by_size.get(s_int, 0) + 1

        # count merges (lost clusters participated in merges)
        for lid in lost:
            idx = np.where(ids_prev==lid)[0]
            if len(idx)==1:
                s = int(round(size_prev[idx[0]]))
                merges_by_size[s] = merges_by_size.get(s, 0) + 1

    return merges_by_size, exists_by_size

# ================================================================
#  PER-SCENARIO PROCESSING
# ================================================================
def process_scenario(
        scen_sim_folder: str,
        scen_out_folder: str,
        bin_width=20.0,
        max_dist=500.0,
        size_tol=2.0):

    os.makedirs(scen_out_folder, exist_ok=True)

    # get repeats
    rep_dirs = sorted(
        d for d in os.listdir(scen_sim_folder)
        if d.startswith("repeat_")
        and os.path.isdir(os.path.join(scen_sim_folder, d))
    )

    bins = np.arange(0, max_dist+bin_width, bin_width)
    centres = 0.5*(bins[:-1]+bins[1:])

    sum_merged = np.zeros(len(centres), int)
    sum_total = np.zeros(len(centres), int)

    # size stats
    merged_size_all = {}
    exists_size_all = {}

    for rep in rep_dirs:
        rep_path = os.path.join(scen_sim_folder, rep)

        # Distance-based stats
        try:
            m, t = repeat_pair_stats(rep_path, bins, size_tol)
            sum_merged += m
            sum_total += t
        except Exception as e:
            print(f"[warn] distance stats failed in {rep}: {e}")

        # Size-based stats
        try:
            msz, esz = repeat_size_stats(rep_path)
            for s,v in msz.items():
                merged_size_all[s] = merged_size_all.get(s,0)+v
            for s,v in esz.items():
                exists_size_all[s] = exists_size_all.get(s,0)+v
        except Exception as e:
            print(f"[warn] size stats failed in {rep}: {e}")

    # --- Probability vs distance ---
    prob = np.where(sum_total>0, sum_merged/sum_total, np.nan)
    df = pd.DataFrame({
        "bin_left": bins[:-1],
        "bin_right": bins[1:],
        "bin_centre": centres,
        "pairs_total": sum_total,
        "pairs_merged": sum_merged,
        "prob_merged": prob
    })
    df.to_csv(os.path.join(scen_out_folder,
               "pairwise_coagulation_vs_initial_distance.csv"), index=False)

    plt.figure(figsize=(8,5))
    plt.plot(centres, prob, marker="o", lw=2)
    plt.xlabel("Initial pair displacement")
    plt.ylabel("P(pair lineages merge)")
    plt.title("Pairwise coagulation vs distance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(scen_out_folder,
                "pairwise_coagulation_vs_initial_distance.png"), dpi=150)
    plt.close()

    # --- Probability vs size ---
    sizes_sorted = sorted(set(list(merged_size_all.keys()) +
                              list(exists_size_all.keys())))

    prob_size = []
    for s in sizes_sorted:
        m = merged_size_all.get(s, 0)
        e = exists_size_all.get(s, 0)
        prob_size.append(m/e if e>0 else np.nan)

    df2 = pd.DataFrame({
        "size": sizes_sorted,
        "exists": [exists_size_all.get(s,0) for s in sizes_sorted],
        "merged": [merged_size_all.get(s,0) for s in sizes_sorted],
        "prob": prob_size
    })
    df2.to_csv(os.path.join(scen_out_folder,
                "coagulation_vs_size.csv"), index=False)

    plt.figure(figsize=(8,5))
    plt.plot(sizes_sorted, prob_size, marker="o", lw=2)
    plt.xlabel("Cluster size")
    plt.ylabel("P(coagulation | size)")
    plt.title("Coagulation probability vs size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(scen_out_folder,
                "coagulation_vs_size.png"), dpi=150)
    plt.close()


# ================================================================
#  LEGEND UTILITIES FOR OVERLAY
# ================================================================
def parse_scenario_name(scen: str) -> dict:
    parts = scen.split("__")[1:]
    out = {}
    for p in parts:
        if "_" not in p: continue
        k,v = p.split("_",1)
        if "p" in v: v = v.replace("p",".")
        try: v = float(v)
        except: pass
        out[k] = v
    return out

def varying_keys(dct: Dict[str,dict]) -> List[str]:
    keys = set()
    for v in dct.values():
        keys |= set(v.keys())
    out=[]
    for k in keys:
        vals={v.get(k,None) for v in dct.values()}
        if len(vals)>1:
            out.append(k)
    return sorted(out)

def build_label(pd: dict, varks: List[str]) -> str:
    parts=[]
    for k in varks:
        v=pd.get(k,None)
        if isinstance(v,float):
            s=f"{v:.6g}".rstrip("0").rstrip(".")
            parts.append(f"{k}={s}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


# ================================================================
#  GLOBAL OVERLAYS: DISTANCE + SIZE
# ================================================================
def overlay_all_scenarios(run_dir: str, scenarios: List[str]):
    extra_lineages_root = os.path.join(run_dir,"extra_plots_lineages")
    out_root = os.path.join(run_dir,"extra_plots")
    os.makedirs(out_root, exist_ok=True)

    scen_params = {s: parse_scenario_name(s) for s in scenarios}
    var_keys = varying_keys(scen_params)

    # ----------------- Distance overlays -----------------
    dist_rows=[]
    for scen in scenarios:
        pth = os.path.join(extra_lineages_root,scen,"pairwise_coagulation_vs_initial_distance.csv")
        if not os.path.exists(pth):
            pth = os.path.join(extra_lineages_root,scen,"pairwise_coagulation_vs_initial_idistance.csv")
            if not os.path.exists(pth):
                continue
        df=pd.read_csv(pth)
        for _,r in df.iterrows():
            dist_rows.append({
                "scenario":scen,
                "d":float(r["bin_centre"]),
                "prob":float(r["prob_merged"])
            })

    if dist_rows:
        dist_df=pd.DataFrame(dist_rows)
        dist_df.to_csv(os.path.join(out_root,
            "pairwise_coagulation_vs_initial_distance_ALL_SCENARIOS.csv"),index=False)

        plt.figure(figsize=(10,6))
        for scen,sub in dist_df.groupby("scenario"):
            lbl = build_label(scen_params[scen], var_keys)
            plt.plot(sub["d"], sub["prob"], marker="o", lw=2, label=lbl if lbl else scen)
        plt.xlabel("Initial displacement")
        plt.ylabel("P(pair lineage merge)")
        plt.title("Pairwise coagulation vs distance (all scenarios)")
        plt.grid(True,alpha=0.3)
        plt.legend(frameon=False,bbox_to_anchor=(1.02,1),loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root,
            "pairwise_coagulation_vs_initial_distance_ALL_SCENARIOS.png"),dpi=160)
        plt.close()

    # ----------------- Size overlays -----------------
    size_rows=[]
    for scen in scenarios:
        pth = os.path.join(extra_lineages_root,scen,"coagulation_vs_size.csv")
        if not os.path.exists(pth):
            continue
        df=pd.read_csv(pth)
        for _,r in df.iterrows():
            size_rows.append({
                "scenario":scen,
                "size":int(r["size"]),
                "prob":float(r["prob"])
            })

    if size_rows:
        size_df=pd.DataFrame(size_rows)
        size_df.to_csv(os.path.join(out_root,
            "coagulation_vs_size_ALL_SCENARIOS.csv"),index=False)

        plt.figure(figsize=(10,6))
        for scen,sub in size_df.groupby("scenario"):
            lbl = build_label(scen_params[scen], var_keys)
            plt.plot(sub["size"], sub["prob"], marker="o", lw=2,
                     label=lbl if lbl else scen)
        plt.xlabel("Cluster size")
        plt.ylabel("P(coagulation|size)")
        plt.title("Coagulation probability vs size (all scenarios)")
        plt.grid(True,alpha=0.3)
        plt.legend(frameon=False,bbox_to_anchor=(1.02,1),loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root,
            "coagulation_vs_size_ALL_SCENARIOS.png"),dpi=160)
        plt.close()


# ================================================================
#  CLI
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bin-width", type=float, default=20.0)
    ap.add_argument("--max-dist", type=float, default=500.0)
    ap.add_argument("--size-tol", type=float, default=2.0)
    args = ap.parse_args()

    run_dir = args.run.rstrip("/")
    sim_root = os.path.join(run_dir,"simulations")
    if not os.path.isdir(sim_root):
        raise RuntimeError(f"Cannot find simulations/ in {run_dir}")

    extra_lineages_root = os.path.join(run_dir,"extra_plots_lineages")
    os.makedirs(extra_lineages_root,exist_ok=True)

    scenarios = sorted(
        d for d in os.listdir(sim_root)
        if os.path.isdir(os.path.join(sim_root,d))
    )

    for scen in scenarios:
        print(f"\n[scenario] {scen}")
        scen_in = os.path.join(sim_root,scen)
        scen_out = os.path.join(extra_lineages_root,scen)
        process_scenario(
            scen_in, scen_out,
            bin_width=args.bin_width,
            max_dist=args.max_dist,
            size_tol=args.size_tol
        )

    overlay_all_scenarios(run_dir, scenarios)


if __name__ == "__main__":
    main()