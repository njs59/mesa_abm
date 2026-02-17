import os
import yaml
import time
from datetime import datetime
from contextlib import contextmanager

def utcnow():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

@contextmanager
def timer():
    start = time.perf_counter()
    obj = type("TimerResult", (), {})()
    try:
        yield obj
    finally:
        obj.seconds = time.perf_counter() - start

def make_run_dir(results_root: str) -> str:
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(results_root, f'run_{ts}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_manifest(run_dir: str, cfg: dict, extra: dict = None):
    man = {'config': cfg}
    if extra:
        man.update(extra)
    path = os.path.join(run_dir, 'pipeline_manifest.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(man, f, sort_keys=False)
    return path

def save_run_input(run_dir: str, cfg: dict):
    snapshot = {
        "run_id": os.path.basename(run_dir),
        "timestamp": utcnow(),
        "abm": cfg.get("abm", {}),
        "mcmc": cfg.get("mcmc", {}),
        "models": cfg.get("models", []),
        "models_meta": cfg.get("models_meta", {}),
        "paths": {
            "results_root": cfg.get("results_root"),
            "run_directory": run_dir,
        },
    }
    path = os.path.join(run_dir, "run_input.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False)
    return path

def write_timings(run_dir: str, timings: dict):
    path = os.path.join(run_dir, "timings.yaml")
    existing = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = yaml.safe_load(f) or {}
    existing.update(timings)
    with open(path, "w") as f:
        yaml.safe_dump(existing, f, sort_keys=False)
    return path

def init_run_status(run_dir: str):
    path = os.path.join(run_dir, "run_status.yaml")
    status = {
        "status": "initialising",
        "last_update": utcnow(),
        "pipeline_total": {"started": utcnow(), "finished": None},
        "sections": {}
    }
    with open(path, "w") as f:
        yaml.safe_dump(status, f, sort_keys=False)
    return path

def update_run_status(run_dir: str, updates: dict):
    path = os.path.join(run_dir, "run_status.yaml")
    try:
        with open(path, "r") as f:
            status = yaml.safe_load(f) or {}
    except Exception:
        status = {}

    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v

    merge(status, updates)
    status["last_update"] = utcnow()

    with open(path, "w") as f:
        yaml.safe_dump(status, f, sort_keys=False)
    return path