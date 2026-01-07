
# -*- coding: utf-8 -*-
import copy
from datetime import datetime
import os
import numpy as np

import solara
from solara.lab import use_task
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS, export_timeseries_state


# ---------------------------------------------------------------
# Reactive state
# ---------------------------------------------------------------
model_r = solara.reactive(None)          # ClustersModel instance
running_r = solara.reactive(False)       # Run/pause
steps_per_second_r = solara.reactive(10) # Simulation speed
step_r = solara.reactive(0)              # Current step counter (UI)
version_r = solara.reactive(0)           # Rerender tick for plots
params_r = solara.reactive(copy.deepcopy(DEFAULTS))

# Figures to enable optional saving (if you want a save bar later)
space_fig_r = solara.reactive(None)
hist_fig_r = solara.reactive(None)
summary_fig_r = solara.reactive(None)


# ---------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------
def make_model(seed: int = 42, params=None, init_clusters=None) -> ClustersModel:
    return ClustersModel(params=params or DEFAULTS, seed=seed, init_clusters=init_clusters)


# ---------------------------------------------------------------
# Background runner (async & persistent)
# ---------------------------------------------------------------
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


# ---------------------------------------------------------------
# Parameter controls
# ---------------------------------------------------------------

@solara.component
def ParamControls():
    p = params_r.value
    # Seed & sim rate
    seed, set_seed = solara.use_state(42)

    # ---- Space ----
    W, set_W = solara.use_state(int(p["space"]["width"]))
    H, set_H = solara.use_state(int(p["space"]["height"]))
    torus, set_torus = solara.use_state(bool(p["space"]["torus"]))

    # ---- Time ----
    dt, set_dt = solara.use_state(float(p["time"]["dt"]))

    # ---- Merge ----
    merge_prob, set_merge_prob = solara.use_state(float(p["merge"]["prob_contact_merge"]))

    # ---- Proliferative phenotype ----
    prolif_speed, set_prolif_speed = solara.use_state(float(p["phenotypes"]["proliferative"]["speed_base"]))
    prolif_rate, set_prolif_rate = solara.use_state(float(p["phenotypes"]["proliferative"]["prolif_rate"]))
    prolif_adh, set_prolif_adh = solara.use_state(float(p["phenotypes"]["proliferative"]["adhesion"]))
    prolif_frag, set_prolif_frag = solara.use_state(float(p["phenotypes"]["proliferative"]["fragment_rate"]))

    # ---- Invasive phenotype ----
    inv_speed, set_inv_speed = solara.use_state(float(p["phenotypes"]["invasive"]["speed_base"]))
    inv_rate, set_inv_rate = solara.use_state(float(p["phenotypes"]["invasive"]["prolif_rate"]))
    inv_adh, set_inv_adh = solara.use_state(float(p["phenotypes"]["invasive"]["adhesion"]))
    inv_frag, set_inv_frag = solara.use_state(float(p["phenotypes"]["invasive"]["fragment_rate"]))

    # ---- Movement options (NEW) ----
    mv_defaults = p.get("movement", {})
    mode_default = str(mv_defaults.get("mode", "constant"))
    dir_default = str(mv_defaults.get("direction", "isotropic"))
    dist_default = str(mv_defaults.get("distribution", "lognorm"))
    dp_defaults = mv_defaults.get("dist_params", {}) or {}

    mode, set_mode = solara.use_state(mode_default)                    # "constant" or "distribution"
    direction, set_direction = solara.use_state(dir_default)           # "isotropic" or "keep"
    dist_name, set_dist_name = solara.use_state(dist_default)          # distribution name
    heading_sigma, set_heading_sigma = solara.use_state(float(mv_defaults.get("heading_sigma", 0.25)))

    # Parameter states per distribution (kept across switches)
    # lognorm: s, scale
    ln_s, set_ln_s = solara.use_state(float(dp_defaults.get("s", 0.6)))
    ln_scale, set_ln_scale = solara.use_state(float(dp_defaults.get("scale", 2.0)))
    # gamma: a, scale
    ga_a, set_ga_a = solara.use_state(float(dp_defaults.get("a", 2.0)))
    ga_scale, set_ga_scale = solara.use_state(float(dp_defaults.get("scale", 1.0)))
    # weibull: c, scale
    wb_c, set_wb_c = solara.use_state(float(dp_defaults.get("c", 1.5)))
    wb_scale, set_wb_scale = solara.use_state(float(dp_defaults.get("scale", 2.0)))
    # rayleigh: scale
    ry_scale, set_ry_scale = solara.use_state(float(dp_defaults.get("scale", 2.0)))
    # expon: scale
    ex_scale, set_ex_scale = solara.use_state(float(dp_defaults.get("scale", 1.0)))
    # invgauss: mu, scale
    ig_mu, set_ig_mu = solara.use_state(float(dp_defaults.get("mu", 1.0)))
    ig_scale, set_ig_scale = solara.use_state(float(dp_defaults.get("scale", 1.0)))

    # ---- Initial conditions ----
    n_prolif, set_n_prolif = solara.use_state(300)
    n_invasive, set_n_invasive = solara.use_state(200)
    size_prolif, set_size_prolif = solara.use_state(1)
    size_invasive, set_size_invasive = solara.use_state(2)

    def apply_params():
        newp = copy.deepcopy(DEFAULTS)
        # Space
        newp["space"]["width"] = float(W)
        newp["space"]["height"] = float(H)
        newp["space"]["torus"] = bool(torus)
        # Time
        newp["time"]["dt"] = float(dt)
        # Merge
        newp["merge"]["prob_contact_merge"] = float(merge_prob)
        # Phenotypes
        newp["phenotypes"]["proliferative"]["speed_base"] = float(prolif_speed)
        newp["phenotypes"]["proliferative"]["prolif_rate"] = float(prolif_rate)
        newp["phenotypes"]["proliferative"]["adhesion"] = float(prolif_adh)
        newp["phenotypes"]["proliferative"]["fragment_rate"] = float(prolif_frag)
        newp["phenotypes"]["invasive"]["speed_base"] = float(inv_speed)
        newp["phenotypes"]["invasive"]["prolif_rate"] = float(inv_rate)
        newp["phenotypes"]["invasive"]["adhesion"] = float(inv_adh)
        newp["phenotypes"]["invasive"]["fragment_rate"] = float(inv_frag)

        # Movement (NEW)
        mv_dict = {
            "mode": str(mode),                    # "constant" or "distribution"
            "direction": str(direction),          # "isotropic" or "keep"
            "heading_sigma": float(heading_sigma)
        }
        if mode == "distribution":
            mv_dict["distribution"] = str(dist_name)
            # Write parameters according to chosen distribution
            if dist_name == "lognorm":
                mv_dict["dist_params"] = {"s": float(ln_s), "scale": float(ln_scale)}
            elif dist_name == "gamma":
                mv_dict["dist_params"] = {"a": float(ga_a), "scale": float(ga_scale)}
            elif dist_name == "weibull":
                mv_dict["dist_params"] = {"c": float(wb_c), "scale": float(wb_scale)}
            elif dist_name == "rayleigh":
                mv_dict["dist_params"] = {"scale": float(ry_scale)}
            elif dist_name == "expon":
                mv_dict["dist_params"] = {"scale": float(ex_scale)}
            elif dist_name == "invgauss":
                mv_dict["dist_params"] = {"mu": float(ig_mu), "scale": float(ig_scale)}
            else:
                # Fallback: keep any typed values but avoid missing keys
                mv_dict["dist_params"] = dp_defaults
        else:
            # constant mode: no dist_params needed, but keep a small empty dict for clarity
            mv_dict["distribution"] = str(dist_name)
            mv_dict["dist_params"] = {}

        newp["movement"] = mv_dict

        params_r.value = newp
        init_clusters = (
            [{"size": int(size_prolif), "phenotype": "proliferative"} for _ in range(int(n_prolif))]
            + [{"size": int(size_invasive), "phenotype": "invasive"} for _ in range(int(n_invasive))]
        )
        model_r.value = make_model(seed=seed, params=newp, init_clusters=init_clusters)
        running_r.value = False
        step_r.value = 0
        version_r.value += 1

    # ----------------- UI -----------------
    with solara.Card("Model parameters"):
        # Space
        solara.Markdown("**Space**")
        with solara.Row():
            solara.SliderInt("Width", value=W, min=50, max=2000, on_value=set_W)
            solara.SliderInt("Height", value=H, min=50, max=2000, on_value=set_H)
            solara.Switch(label="Torus", value=torus, on_value=set_torus)

        # Time
        solara.Markdown("**Time**")
        solara.SliderFloat("dt (minutes)", value=dt, min=0.01, max=10.0, step=0.01, on_value=set_dt)

        # Merge
        solara.Markdown("**Merge**")
        solara.SliderFloat("prob_contact_merge", value=merge_prob, min=0.0, max=1.0, step=0.01, on_value=set_merge_prob)

        # Proliferative phenotype
        solara.Markdown("**Proliferative phenotype**")
        with solara.Row():
            solara.SliderFloat("speed_base", value=prolif_speed, min=0.0, max=20.0, step=0.1, on_value=set_prolif_speed)
            solara.SliderFloat("prolif_rate", value=prolif_rate, min=0.0, max=0.05, step=0.0005, on_value=set_prolif_rate)
        with solara.Row():
            solara.SliderFloat("adhesion", value=prolif_adh, min=0.0, max=1.0, step=0.01, on_value=set_prolif_adh)
            solara.SliderFloat("fragment_rate", value=prolif_frag, min=0.0, max=0.05, step=0.0005, on_value=set_prolif_frag)

        # Invasive phenotype
        solara.Markdown("**Invasive phenotype**")
        with solara.Row():
            solara.SliderFloat("speed_base", value=inv_speed, min=0.0, max=20.0, step=0.1, on_value=set_inv_speed)
            solara.SliderFloat("prolif_rate", value=inv_rate, min=0.0, max=0.05, step=0.0005, on_value=set_inv_rate)
        with solara.Row():
            solara.SliderFloat("adhesion", value=inv_adh, min=0.0, max=1.0, step=0.01, on_value=set_inv_adh)
            solara.SliderFloat("fragment_rate", value=inv_frag, min=0.0, max=0.05, step=0.0005, on_value=set_inv_frag)

        # Movement (NEW)
        solara.Markdown("**Movement**")
        with solara.Row():
            solara.Select("Mode", value=mode, values=["constant", "distribution"], on_value=set_mode)
            solara.Select("Direction", value=direction, values=["isotropic", "persistent"], on_value=set_direction)
            solara.SliderFloat("heading_sigma (rad)", value=heading_sigma, min=0.0, max=1.0, step=0.01, on_value=set_heading_sigma)

        if mode == "distribution":
            solara.Markdown("*Distribution parameters*")
            solara.Select("Distribution", value=dist_name,
                          values=["lognorm", "gamma", "weibull", "rayleigh", "expon", "invgauss"],
                          on_value=set_dist_name)
            if dist_name == "lognorm":
                with solara.Row():
                    solara.SliderFloat("lognorm: s (shape)", value=ln_s, min=0.01, max=50.0, step=0.01, on_value=set_ln_s)
                    solara.SliderFloat("lognorm: scale", value=ln_scale, min=0.01, max=50.0, step=0.01, on_value=set_ln_scale)
            elif dist_name == "gamma":
                with solara.Row():
                    solara.SliderFloat("gamma: a (shape)", value=ga_a, min=0.01, max=10.0, step=0.01, on_value=set_ga_a)
                    solara.SliderFloat("gamma: scale", value=ga_scale, min=0.01, max=50.0, step=0.01, on_value=set_ga_scale)
            elif dist_name == "weibull":
                with solara.Row():
                    solara.SliderFloat("weibull: c (shape)", value=wb_c, min=0.01, max=10.0, step=0.01, on_value=set_wb_c)
                    solara.SliderFloat("weibull: scale", value=wb_scale, min=0.01, max=50.0, step=0.01, on_value=set_wb_scale)
            elif dist_name == "rayleigh":
                solara.SliderFloat("rayleigh: scale", value=ry_scale, min=0.01, max=50.0, step=0.01, on_value=set_ry_scale)
            elif dist_name == "expon":
                solara.SliderFloat("expon: scale", value=ex_scale, min=0.01, max=50.0, step=0.01, on_value=set_ex_scale)
            elif dist_name == "invgauss":
                with solara.Row():
                    solara.SliderFloat("invgauss: mu", value=ig_mu, min=0.01, max=50.0, step=0.01, on_value=set_ig_mu)
                    solara.SliderFloat("invgauss: scale", value=ig_scale, min=0.01, max=50.0, step=0.01, on_value=set_ig_scale)

        # Initial condition
        solara.Markdown("**Initial clusters**")
        with solara.Row():
            solara.SliderInt("Proliferative: count", value=n_prolif, min=0, max=1000, on_value=set_n_prolif)
            solara.SliderInt("Proliferative: base size", value=size_prolif, min=1, max=100, on_value=set_size_prolif)
        with solara.Row():
            solara.SliderInt("Invasive: count", value=n_invasive, min=0, max=1000, on_value=set_n_invasive)
            solara.SliderInt("Invasive: base size", value=size_invasive, min=1, max=100, on_value=set_size_invasive)

        # Simulation setup
        solara.Markdown("**Simulation setup**")
        with solara.Row():
            solara.InputInt("Random seed", value=seed, on_value=set_seed)
            solara.SliderInt(
                "Steps/second",
                value=steps_per_second_r.value, min=1, max=120,
                on_value=lambda v: steps_per_second_r.set(v),
            )
        solara.Button("Apply parameters & Reset", color="primary", on_click=apply_params, icon_name="mdi-restart")


# ---------------------------------------------------------------
# Controls
# ---------------------------------------------------------------
@solara.component
def Controls():
    def on_step():
        if model_r.value is None:
            return
        model_r.value.step()
        step_r.value += 1
        version_r.value += 1

    def on_export():
        if model_r.value is None:
            return
        os.makedirs("results", exist_ok=True)
        export_timeseries_state(model_r.value, out_csv="results/state_timeseries.csv")

    with solara.Card("Simulation controls"):
        with solara.Row(gap="1rem"):
            if running_r.value:
                solara.Button("Pause", color="primary",
                              on_click=lambda: running_r.set(False), icon_name="mdi-pause")
            else:
                solara.Button("Run", color="primary",
                              on_click=lambda: running_r.set(True), icon_name="mdi-play")
            solara.Button("Step", on_click=on_step, icon_name="mdi-skip-next")
            solara.Button("Export CSV", on_click=on_export, icon_name="mdi-file-export")
            solara.Text(f"Step: {step_r.value}")


# ---------------------------------------------------------------
# Space View (draw discs from wrapped coordinates stored in agent.pos)
# ---------------------------------------------------------------
@solara.component
def SpaceView():
    m = model_r.value
    if m is None:
        return solara.Warning("Model not initialised yet.")

    all_agents = list(m.agents)
    valid_agents = []
    none_pos = 0
    for a in all_agents:
        if not getattr(a, "alive", True):
            continue
        p = getattr(a, "pos", None)
        if p is None:
            none_pos += 1
            continue
        try:
            float(p[0]); float(p[1])
        except Exception:
            none_pos += 1
            continue
        valid_agents.append(a)

    solara.Info(f"Agents total: {len(all_agents)}, drawable: {len(valid_agents)}, pos=None: {none_pos}")

    fig = Figure()
    ax = fig.subplots()

    for a in valid_agents:
        color = m.params["phenotypes"][a.phenotype]["color"]
        x, y = float(a.pos[0]), float(a.pos[1])  # ContinuousSpace maintains wrapped .pos
        r = float(a.radius)
        ax.add_patch(Circle((x, y), radius=r, facecolor=color, edgecolor="none", alpha=0.70))

    W = float(m.params["space"]["width"])
    H = float(m.params["space"]["height"])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
    ax.set_title("Clusters (continuous space)")
    ax.grid(True, alpha=0.2)

    space_fig_r.value = fig
    return solara.FigureMatplotlib(fig, dependencies=[version_r.value])


# ---------------------------------------------------------------
# Size histogram (live snapshot; avoids log race conditions)
# ---------------------------------------------------------------
@solara.component
def SizeHistogram():
    m = model_r.value
    if m is None:
        return solara.Warning("Model not initialised yet.")

    alive_now = [
        a for a in m.agents
        if getattr(a, "alive", True) and getattr(a, "pos", None) is not None
    ]
    sizes = np.asarray([getattr(a, "size", 0) for a in alive_now], dtype=float)

    if sizes.size == 0:
        return solara.Warning("No clusters to display yet.")

    max_size = int(np.max(sizes))
    bins = np.arange(1, max_size + 2) if max_size >= 1 else np.arange(1, 3)

    fig = Figure()
    ax = fig.subplots()
    ax.hist(sizes, bins=bins, color="#3f51b5")
    ax.set_title("Cluster size distribution (latest)")
    ax.set_xlabel("Size (cells)")
    ax.set_ylabel("Count")

    hist_fig_r.value = fig
    return solara.FigureMatplotlib(fig, dependencies=[version_r.value])


# ---------------------------------------------------------------
# Save Figures bar
# ---------------------------------------------------------------
@solara.component
def SaveFiguresBar():
    base_name, set_base_name = solara.use_state("figure")
    fmt, set_fmt = solara.use_state("png")  # 'png' or 'svg'
    dpi, set_dpi = solara.use_state(150)
    add_ts, set_add_ts = solara.use_state(True)
    msg_r = solara.use_reactive("")

    def timestamp():
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def ensure_dir(path):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def save_figure(fig, kind: str):
        if fig is None:
            msg_r.value = f"No {kind} figure available yet."
            return
        ts = f"_{timestamp()}" if add_ts else ""
        filename = f"figures/{base_name}_{kind}{ts}.{fmt}"
        ensure_dir(filename)
        try:
            if fmt.lower() == "png":
                fig.savefig(filename, dpi=int(dpi), bbox_inches="tight")
            else:
                fig.savefig(filename, bbox_inches="tight")
            msg_r.value = f"Saved {kind} to '{filename}'."
        except Exception as e:
            msg_r.value = f"Failed to save {kind}: {e}"

    with solara.Card("Save figures"):
        with solara.Row(gap="1rem"):
            solara.InputText("Base name", value=base_name, on_value=set_base_name)
            solara.Select("Format", value=fmt, values=["png", "svg"], on_value=set_fmt)
            solara.SliderInt("DPI (PNG only)", value=dpi, min=72, max=600, on_value=set_dpi)
            solara.Switch(label="Include timestamp", value=add_ts, on_value=set_add_ts)
        with solara.Row(gap="1rem"):
            solara.Button(
                "Save Space view",
                icon_name="mdi-content-save",
                color="primary",
                on_click=lambda: save_figure(space_fig_r.value, "space"),
            )
            solara.Button(
                "Save Size histogram",
                icon_name="mdi-content-save",
                color="primary",
                on_click=lambda: save_figure(hist_fig_r.value, "hist"),
            )
            solara.Button(
                               "Save Summary",
                icon_name="mdi-content-save",
                color="primary",
                on_click=lambda: save_figure(summary_fig_r.value, "summary"),
            )
        if msg_r.value:
            solara.Success(msg_r.value)

# ---------------------------------------------------------------
# Summary plots (live): N(t), mean size ± SEM(t), total cells(t)
# ---------------------------------------------------------------
@solara.component
def SummaryPlots():
    m = model_r.value
    if m is None:
        return solara.Warning("Model not initialised yet.")

    # Snapshot the log references once to avoid mid-render mutations
    size_log = list(m.size_log)

    if len(size_log) == 0:
        # Fallback to a single-point series from the live snapshot
        alive_now = [a for a in m.agents if getattr(a, "alive", True) and getattr(a, "pos", None) is not None]
        sizes_now = np.asarray([getattr(a, "size", 0) for a in alive_now], dtype=float)
        steps = np.array([0], dtype=int)
        n_series = np.array([len(sizes_now)], dtype=int)
        mean_series = np.array([float(np.mean(sizes_now)) if sizes_now.size else 0.0], dtype=float)
        sem_series = np.array([float(np.std(sizes_now, ddof=1) / np.sqrt(len(sizes_now))) if len(sizes_now) > 1 else 0.0], dtype=float)
        total_series = np.array([float(np.sum(sizes_now))], dtype=float)
    else:
        steps = np.arange(len(size_log), dtype=int)
        n_series = np.asarray([len(s) for s in size_log], dtype=int)
        mean_series = np.asarray([float(np.mean(s)) if len(s) > 0 else 0.0 for s in size_log], dtype=float)
        sem_series = np.asarray([
            float(np.std(s, ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0
            for s in size_log
        ], dtype=float)
        total_series = np.asarray([float(np.sum(s)) if len(s) > 0 else 0.0 for s in size_log], dtype=float)

    # Windowed view for performance
    default_window = 500
    win, set_win = solara.use_state(default_window)
    start_idx = max(0, len(steps) - int(win))
    sl = slice(start_idx, len(steps))

    s_steps = steps[sl]
    s_n = n_series[sl]
    s_mean = mean_series[sl]
    s_sem = sem_series[sl]
    s_total = total_series[sl]

    # Live snapshot metrics (always computed from agents to match SpaceView)
    alive_now = [a for a in m.agents if getattr(a, "alive", True) and getattr(a, "pos", None) is not None]
    sizes_now = np.asarray([getattr(a, "size", 0) for a in alive_now], dtype=float)
    n_now = int(len(sizes_now))
    mean_now = float(np.mean(sizes_now)) if n_now > 0 else 0.0
    sem_now = float(np.std(sizes_now, ddof=1) / np.sqrt(n_now)) if n_now > 1 else 0.0
    total_now = float(np.sum(sizes_now)) if n_now > 0 else 0.0

    dt = float(params_r.value["time"]["dt"])

    with solara.Card("Summary (live)"):
        with solara.Row(gap="1rem"):
            solara.Text(f"Step: {step_r.value} | t = {step_r.value * dt:.1f} min")
            solara.Text(f"N clusters: {n_now}")
            solara.Text(f"Mean size: {mean_now:.2f} ± {sem_now:.2f} (SEM)")
            solara.Text(f"Total cells: {total_now:.0f}")

        with solara.Row(gap="1rem"):
            solara.SliderInt("Show last N steps", value=win, min=50, max=5000, on_value=set_win)

        fig = Figure(figsize=(7.5, 6.5))
        axs = fig.subplots(3, 1, sharex=True)

        # 1) Number of clusters
        axs[0].plot(s_steps, s_n, color="tab:blue", lw=2)
        axs[0].set_ylabel("Number of\nclusters")
        axs[0].grid(True, alpha=0.2)

        # 2) Mean size ± SEM
        axs[1].plot(s_steps, s_mean, color="tab:green", lw=2, label="Mean size")
        axs[1].fill_between(s_steps, s_mean - s_sem, s_mean + s_sem, color="tab:green", alpha=0.2, label="± SEM")
        axs[1].set_ylabel("Mean size\n(cells)")
        axs[1].legend(loc="upper left", frameon=False)
        axs[1].grid(True, alpha=0.2)

        # 3) Total cells
        axs[2].plot(s_steps, s_total, color="tab:purple", lw=2)
        axs[2].set_ylabel("Total cells")
        axs[2].set_xlabel("Step")
        axs[2].grid(True, alpha=0.2)

        fig.tight_layout()
        summary_fig_r.value = fig
        return solara.FigureMatplotlib(fig, dependencies=[version_r.value])


# ---------------------------------------------------------------
# Page
# ---------------------------------------------------------------
@solara.component
def Page():
    # Initialise model once and trigger an initial render
    def on_mount():
        if model_r.value is None:
            model_r.set(make_model(seed=42))
        version_r.value += 1

    _ = solara.use_effect(on_mount, [])  # run once after mount

    with solara.Column(gap="1.0rem"):
        ParamControls()
        Controls()
        Runner()
        with solara.Row(gap="2rem"):
            SpaceView()
            SizeHistogram()
            SaveFiguresBar()
        SummaryPlots()
