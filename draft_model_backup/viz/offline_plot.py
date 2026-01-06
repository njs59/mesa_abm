
from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS
import copy
import matplotlib.pyplot as plt

if __name__ == "__main__":
    params = copy.deepcopy(DEFAULTS)
    params["space"]["torus"] = True
    model = ClustersModel(params=params, seed=123)
    for _ in range(int(params["time"]["steps"])):
        model.step()

    fig, ax = plt.subplots(figsize=(6, 6))
    W = params["space"]["width"]; H = params["space"]["height"]
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal')
    ax.set_title("ABM Snapshot — periodic boundaries (torus=True)")
    for a in model.agents:
        if not a.alive or a.pos is None:
            continue
        color = DEFAULTS["phenotypes"][a.phenotype]["color"]
        x, y = a.pos
        circ = plt.Circle((x, y), a.radius, fc=color, ec='k', alpha=0.6)
        ax.add_patch(circ)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    fig.tight_layout()
    import os; os.makedirs('results', exist_ok=True)
    fig.savefig('results/snapshot_final.png', dpi=150)
    print('Saved results/snapshot_final.png')
