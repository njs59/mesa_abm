#!/usr/bin/env python3
"""
Summarise existing pyABC SQLite DBs into a CSV, with flexible selection of .db files.

You can:
  1) Provide explicit .db paths (one or many), OR
  2) Provide one or more glob patterns, OR
  3) Scan a root directory recursively (use them all).

The script merges all sources, de-duplicates, then parses dataset/motion/speed_dist/seed
from filenames using a format-style pattern, and reads run stats from each DB.

Default filename pattern (matches run_motion_grid.py outputs):
    {dataset}_{motion}_{speed_dist}_s{seed}.db
Examples:
    INV_isotropic_gamma_s42.db
    PRO_persistent_lognorm_s123.db

Outputs CSV columns:
    dataset, motion, speed_dist, seed, final_eps, n_populations, db

Usage examples:
    # Use ALL db files under a root
    python summarise_existing_runs_select.py \
      --root motiongrid_pkg/results \
      --out motiongrid_pkg/results/summary.csv

    # Use only explicit files
    python summarise_existing_runs_select.py \
      --db motiongrid_pkg/results/INV_isotropic_gamma_s42.db \
      --db motiongrid_pkg/results/PRO_isotropic_gamma_s42.db \
      --out summary_gamma_only.csv

    # Use glob patterns
    python summarise_existing_runs_select.py \
      --glob "motiongrid_pkg/results/*_isotropic_*_s*.db" \
      --out summary_isotropic.csv

    # Mix sources (explicit files + globs + full scan)
    python summarise_existing_runs_select.py \
      --root motiongrid_pkg/results \
      --glob "motiongrid_pkg/results/*_persistent_*_s*.db" \
      --db motiongrid_pkg/results/INV_isotropic_gamma_s42.db \
      --out summary_mixed.csv
"""
import argparse
import re
from pathlib import Path
import glob
import pandas as pd
import pyabc

DEFAULT_PATTERN = r"{dataset}_{motion}_{speed_dist}_s{seed}.db"


def make_regex_from_pattern(pattern: str) -> re.Pattern:
    esc = re.escape(pattern)
    subs = {
        re.escape("{dataset}"): r"(?P<dataset>[^_/\\-]+)",
        re.escape("{motion}"): r"(?P<motion>[^_/\\-]+)",
        re.escape("{speed_dist}"): r"(?P<speed_dist>[^_/\\-]+)",
        re.escape("{seed}"): r"(?P<seed>\d+)",
    }
    for k, v in subs.items():
        esc = esc.replace(k, v)
    return re.compile(r"^" + esc + r"$")


def parse_from_name(name: str, pat: re.Pattern) -> dict:
    m = pat.match(name)
    if not m:
        return {}
    g = m.groupdict()
    if "seed" in g and g["seed"] is not None:
        try:
            g["seed"] = int(g["seed"]) 
        except Exception:
            pass
    return g


def summarise_db(db_path: Path) -> dict:
    hist = pyabc.History(f"sqlite:///{db_path.resolve()}")
    try:
        pops = hist.get_all_populations()
        n_pops = int(pops.shape[0])
        final_eps = float(pops.iloc[-1]["epsilon"]) if n_pops > 0 else float("nan")
    except Exception:
        n_pops = int(getattr(hist, "n_populations", 0))
        final_eps = float("nan")
    return {"final_eps": final_eps, "n_populations": n_pops}


def collect_db_paths(root: Path, globs: list[str], explicit: list[str], include_hidden: bool) -> list[Path]:
    paths = set()
    # 1) root scan
    if root is not None:
        for p in root.rglob("*.db"):
            if not include_hidden and any(part.startswith('.') for part in p.parts):
                continue
            paths.add(p.resolve())
    # 2) glob patterns
    for gpat in globs or []:
        for match in glob.glob(gpat, recursive=True):
            p = Path(match)
            if p.suffix == ".db":
                if not include_hidden and any(part.startswith('.') for part in p.parts):
                    continue
                paths.add(p.resolve())
    # 3) explicit files
    for f in explicit or []:
        p = Path(f)
        if p.exists() and p.suffix == ".db":
            paths.add(p.resolve())
    return sorted(paths)


def main():
    ap = argparse.ArgumentParser(description="Summarise pyABC DBs with flexible selection (all or chosen)")
    ap.add_argument("--root", type=str, default=None, help="Directory to scan recursively for .db files (use all)")
    ap.add_argument("--glob", dest="globs", nargs="*", default=None, help="One or more glob patterns to select .db files")
    ap.add_argument("--db", dest="dbs", nargs="*", default=None, help="Explicit .db paths to include")
    ap.add_argument("--out", type=str, default="summary.csv", help="Output CSV path")
    ap.add_argument("--pattern", type=str, default=DEFAULT_PATTERN, help="Filename pattern to parse fields")
    ap.add_argument("--include-hidden", action="store_true", help="Include hidden files/directories in selection")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else None
    paths = collect_db_paths(root, args.globs, args.dbs, args.include_hidden)

    if not paths:
        print("[WARN] No .db files selected. Provide --root, --glob, or --db.")
        # Still write an empty CSV with headers for consistency
        pd.DataFrame(columns=["dataset","motion","speed_dist","seed","final_eps","n_populations","db"]).to_csv(args.out, index=False)
        return

    pat = make_regex_from_pattern(args.pattern)
    rows = []
    for p in paths:
        parsed = parse_from_name(p.name, pat)
        ds = parsed.get("dataset")
        motion = parsed.get("motion")
        speed = parsed.get("speed_dist")
        seed = parsed.get("seed")
        stats = summarise_db(p)
        # db path relative to root if provided, else relative to current dir
        rel = p
        if root:
            try:
                rel = p.relative_to(root)
            except Exception:
                rel = p.name
        else:
            rel = p.name
        rows.append({
            "dataset": ds,
            "motion": motion,
            "speed_dist": speed,
            "seed": seed,
            "final_eps": stats["final_eps"],
            "n_populations": stats["n_populations"],
            "db": str(rel),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[DONE] Wrote summary: {Path(args.out).resolve()} ({len(df)} rows)")


if __name__ == "__main__":
    main()
