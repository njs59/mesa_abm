# -*- coding: utf-8 -*-
"""
Solara UI for the updated two‑phase ABM with shifted‑Gompertz transitions.
Includes phase‑indicator visuals in SpaceView().
"""

import os
import copy
import numpy as np
import solara
from solara.lab import use_task
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from abm.clusters_model import ClustersModel
from abm.utils import DEFAULTS, export_timeseries_state


# -------------------------------------------------------------------
# Reactive state
# -------------------------------------------------------------------
model_r = solara.reactive(None)
running_r = solara.reactive(False)
step_r = solara.reactive(0)
version_r = solara.reactive(0)
steps_per_second_r = solara.reactive(10)

params_r = solara.reactive(copy.deepcopy(DEFAULTS))

space_fig_r = solara.reactive(None)
hist_fig_r = solara.reactive(None)
summary_fig_r = solara.reactive(None)


# -------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------
def make_model(seed=42, params=None, init_clusters=None):
    return ClustersModel(params=params or DEFAULTS, seed=seed, init_clusters=init_clusters)


# -------------------------------------------------------------------
# Background runner
# -------------------------------------------------------------------
@solara.component
def Runner():
    import asyncio

    async def run_async():
        while True:
            if running_r.value and model_r.value is not None:
                interval = 1.0 / max(1, int(steps_per_second_r.value))
                model_r.value.step()
                step_r.value += 1
                version_r.value += 1
                await asyncio.sleep(interval)
            else:
                await asyncio.sleep(0.05)

    use_task(run_async)
    return None


# -------------------------------------------------------------------
# Parameter panel
# -------------------------------------------------------------------
@solara.component
def ParamControls():
    p = params_r.value
    seed, set_seed = solara.use_state(42)

    W, set_W = solara.use_state(int(p["space"]["width"]))
    H, set_H = solara.use_state(int(p["space"]["height"]))
    torus, set_torus = solara.use_state(bool(p["space"]["torus"]))

    dt, set_dt = solara.use_state(float(p["time"]["dt"]))

    merge_prob, set_merge_prob = solara.use_state(float(p["merge"]["p_merge"]))

    # phenotype params
    prolif_speed, set_prolif_speed = solara.use_state(p["phenotypes"]["proliferative"]["speed_base"])
    prolif_rate, set_prolif_rate = solara.use_state(p["phenotypes"]["proliferative"]["prolif_rate"])
    prolif_frag, set_prolif_frag = solara.use_state(p["phenotypes"]["proliferative"]["fragment_rate"])

    inv_speed, set_inv_speed = solara.use_state(p["phenotypes"]["invasive"]["speed_base"])
    inv_rate, set_inv_rate = solara.use_state(p["phenotypes"]["invasive"]["prolif_rate"])
    inv_frag, set_inv_frag = solara.use_state(p["phenotypes"]["invasive"]["fragment_rate"])

    n_prolif, set_n_prolif = solara.use_state(300)
    n_inv, set_n_inv = solara.use_state(200)

    def apply_params():
        newp = copy.deepcopy(DEFAULTS)

        newp["space"]["width"] = float(W)
        newp["space"]["height"] = float(H)
        newp["space"]["torus"] = bool(torus)

        newp["time"]["dt"] = float(dt)
        newp["merge"]["p_merge"] = float(merge_prob)

        newp["phenotypes"]["proliferative"]["speed_base"] = float(prolif_speed)
        newp["phenotypes"]["proliferative"]["prolif_rate"] = float(prolif_rate)
        newp["phenotypes"]["proliferative"]["fragment_rate"] = float(prolif_frag)

        newp["phenotypes"]["invasive"]["speed_base"] = float(inv_speed)
        newp["phenotypes"]["invasive"]["prolif_rate"] = float(inv_rate)
        newp["phenotypes"]["invasive"]["fragment_rate"] = float(inv_frag)

        init_clusters = (
            [{"size": 1, "phenotype": "proliferative"} for _ in range(int(n_prolif))] +
            [{"size": 1, "phenotype": "invasive"} for _ in range(int(n_inv))]
        )

        params_r.value = newp
        model_r.value = make_model(seed=seed, params=newp, init_clusters=init_clusters)
        running_r.value = False
        step_r.value = 0
        version_r.value += 1

    with solara.Card("Model parameters"):
        solara.Markdown("### Space")
        with solara.Row():
            solara.SliderInt("Width", value=W, min=50, max=2000, on_value=set_W)
            solara.SliderInt("Height", value=H, min=50, max=2000, on_value=set_H)
            solara.Switch(label="Torus", value=torus, on_value=set_torus)

        solara.Markdown("### Time")
        solara.SliderFloat("dt (min)", value=dt, min=0.01, max=10, step=0.01, on_value=set_dt)

        solara.Markdown("### Merge")
        solara.SliderFloat("p_merge", value=merge_prob, min=0, max=1, step=0.01, on_value=set_merge_prob)

        solara.Markdown("### Proliferative phenotype")
        with solara.Row():
            solara.SliderFloat("speed_base", value=prolif_speed, min=0, max=20, step=0.1, on_value=set_prolif_speed)
            solara.SliderFloat("prolif_rate", value=prolif_rate, min=0, max=0.05, step=0.0005, on_value=set_prolif_rate)
            solara.SliderFloat("fragment_rate", value=prolif_frag, min=0, max=0.05, step=0.0005, on_value=set_prolif_frag)

        solara.Markdown("### Invasive phenotype")
        with solara.Row():
            solara.SliderFloat("speed_base", value=inv_speed, min=0, max=20, step=0.1, on_value=set_inv_speed)
            solara.SliderFloat("prolif_rate", value=inv_rate, min=0, max=0.05, step=0.0005, on_value=set_inv_rate)
            solara.SliderFloat("fragment_rate", value=inv_frag, min=0, max=0.05, step=0.0005, on_value=set_inv_frag)

        solara.Markdown("### Initial clusters")
        with solara.Row():
            solara.SliderInt("proliferative count", value=n_prolif, min=0, max=2000, on_value=set_n_prolif)
            solara.SliderInt("invasive count", value=n_inv, min=0, max=2000, on_value=set_n_inv)

        solara.Markdown("### Run setup")
        with solara.Row():
            solara.InputInt("Seed", value=seed, on_value=set_seed)
            solara.SliderInt(
                "Steps/sec",
                value=steps_per_second_r.value,
                min=1, max=120,
                on_value=lambda v: steps_per_second_r.set(v),
            )

        solara.Button(
            "Apply parameters & Reset",
            color="primary",
            icon_name="mdi-restart",
            on_click=apply_params,
        )


# -------------------------------------------------------------------
# Controls
# -------------------------------------------------------------------
@solara.component
def Controls():
    def step_once():
        if model_r.value:
            model_r.value.step()
            step_r.value += 1
            version_r.value += 1

    def export_csv():
        if model_r.value:
            os.makedirs("results", exist_ok=True)
            export_timeseries_state(model_r.value, "results/state_timeseries.csv")

    with solara.Card("Simulation controls"):
        with solara.Row():
            if running_r.value:
                solara.Button("Pause", icon_name="mdi-pause", color="primary",
                              on_click=lambda: running_r.set(False))
            else:
                solara.Button("Run", icon_name="mdi-play", color="primary",
                              on_click=lambda: running_r.set(True))

            solara.Button("Step", icon_name="mdi-skip-next", on_click=step_once)
            solara.Button("Export CSV", icon_name="mdi-file-export", on_click=export_csv)

        solara.Text(f"Step: {step_r.value}")


# -------------------------------------------------------------------
# Space view (WITH MOVEMENT-PHASE VISUAL)
# -------------------------------------------------------------------
@solara.component
def SpaceView():
    m = model_r.value
    if m is None:
        return solara.Warning("Model not initialised.")

    # safe snapshot — protects against mutation while iterating
    agents_snapshot = list(m.agents)

    fig = Figure()
    ax = fig.subplots()

    for a in agents_snapshot:
        if not getattr(a, "alive", True):
            continue
        if a.pos is None:
            continue

        base_color = m.params["phenotypes"][a.phenotype]["color"]
        x, y = a.pos

        # main body
        ax.add_patch(
            Circle(
                (x, y),
                radius=a.radius,
                facecolor=base_color,
                edgecolor="none",
                alpha=0.7,
            )
        )

        # --- phase visual overlay ---
        if a.movement_phase == 1:
            phase_color = "yellow"
        else:
            phase_color = "deepskyblue"

        ax.add_patch(
            Circle(
                (x, y),
                radius=a.radius * 0.9,
                fill=False,
                linewidth=1.6,
                edgecolor=phase_color,
                alpha=0.9,
            )
        )

    ax.text(
        0.01,
        1.02,
        "Phase 1 = yellow ring | Phase 2 = blue ring",
        transform=ax.transAxes,
        fontsize=9,
    )

    ax.set_xlim(0, m.params["space"]["width"])
    ax.set_ylim(0, m.params["space"]["height"])
    ax.set_aspect("equal")
    ax.set_title("Clusters (Phase visual enabled)")
    ax.grid(alpha=0.2)

    space_fig_r.value = fig
    return solara.FigureMatplotlib(fig, dependencies=[version_r.value])


# -------------------------------------------------------------------
# Histogram
# -------------------------------------------------------------------
@solara.component
def SizeHistogram():
    m = model_r.value
    if m is None:
        return solara.Warning("Model not initialised.")

    sizes = np.array([a.size for a in list(m.agents) if a.alive and a.pos is not None])
    if sizes.size == 0:
        return solara.Warning("No clusters.")

    bins = np.arange(1, int(sizes.max()) + 2)

    fig = Figure()
    ax = fig.subplots()
    ax.hist(sizes, bins=bins, color="#3f51b5")
    ax.set_title("Cluster size distribution")
    ax.set_xlabel("Size (cells)")
    ax.set_ylabel("Count")

    hist_fig_r.value = fig
    return solara.FigureMatplotlib(fig, dependencies=[version_r.value])


# -------------------------------------------------------------------
# Summary plots
# -------------------------------------------------------------------
@solara.component
def SummaryPlots():
    m = model_r.value
    if m is None:
        return solara.Warning("Model not initialised.")

    size_log = m.size_log

    if len(size_log) == 0:
        sizes = np.array([a.size for a in list(m.agents) if a.alive and a.pos is not None])
        steps = np.array([0])
        n_series = np.array([len(sizes)])
        mean_series = np.array([sizes.mean() if len(sizes) else 0])
        sem_series = np.array([sizes.std(ddof=1) / np.sqrt(len(sizes))
                               if len(sizes) > 1 else 0])
        total_series = np.array([sizes.sum()])
    else:
        steps = np.arange(len(size_log))
        n_series = np.array([len(s) for s in size_log])
        mean_series = np.array([s.mean() for s in size_log])
        sem_series = np.array([
            s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0
            for s in size_log
        ])
        total_series = np.array([s.sum() for s in size_log])

    fig = Figure(figsize=(7, 6))
    axs = fig.subplots(3, 1, sharex=True)

    axs[0].plot(steps, n_series, color="tab:blue")
    axs[0].set_ylabel("N clusters")
    axs[0].grid(alpha=0.2)

    axs[1].plot(steps, mean_series, color="tab:green")
    axs[1].fill_between(
        steps, mean_series - sem_series, mean_series + sem_series, alpha=0.2
    )
    axs[1].set_ylabel("Mean size")
    axs[1].grid(alpha=0.2)

    axs[2].plot(steps, total_series, color="tab:purple")
    axs[2].set_ylabel("Total cells")
    axs[2].set_xlabel("Step")
    axs[2].grid(alpha=0.2)

    fig.tight_layout()
    summary_fig_r.value = fig

    return solara.FigureMatplotlib(fig, dependencies=[version_r.value])


# -------------------------------------------------------------------
# Page root
# -------------------------------------------------------------------
@solara.component
def Page():

    def on_mount():
        if model_r.value is None:
            model_r.set(make_model(seed=42))
            version_r.value += 1

    solara.use_effect(on_mount, [])

    with solara.Column(gap="1rem"):
        ParamControls()
        Controls()
        Runner()

        with solara.Row(gap="2rem"):
            SpaceView()
            SizeHistogram()

        SummaryPlots()