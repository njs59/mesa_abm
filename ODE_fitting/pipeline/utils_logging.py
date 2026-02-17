import os
import yaml
from datetime import datetime


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