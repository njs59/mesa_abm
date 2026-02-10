from mesa import Model
from mesa.space import ContinuousSpace
import numpy as np
from .cluster_agent import ClusterAgent
from .utils import DEFAULTS

class ClustersModel(Model):
    def __init__(self, params=None, seed=42, init_clusters=None):
        super().__init__()
        self.random.seed(seed)
        self.params = params or DEFAULTS
        W=float(self.params["space"]["width"]); H=float(self.params["space"]["height"])
        self.space = ContinuousSpace(x_max=W, y_max=H, torus=bool(self.params["space"]["torus"]))
        self.dt=float(self.params["time"]["dt"])
        self.time=0.0
        if init_clusters is None:
            init_cfg=self.params.get("init", {})
            n0=int(init_cfg.get("n_clusters", 1000))
            sz0=int(init_cfg.get("size", 1))
            ph0=str(init_cfg.get("phenotype", "proliferative"))
            init_clusters=[{"size": sz0, "phenotype": ph0} for _ in range(n0)]
        for ic in init_clusters:
            self.spawn_cluster(size=ic["size"], phenotype=ic["phenotype"], movement_phases=ic.get("movement_phases", None))
        self.size_log=[]; self.speed_log=[]; self.pos_log=[]; self.radius_log=[]; self.id_log=[]
        self._log_state()

    def _maybe_build_phases_for_new_agent(self):
        cfg=self.params.get("movement_phase_builder", None)
        if not cfg: return None
        scen=cfg.get("scenario", "")
        if scen=="two_phase_fixed":
            t_sw=float(cfg["t_switch"])  
            ph1={"t_start":0.0, "direction":"von_mises", "mu": float(cfg.get("mu1", 0.0)), "kappa": float(cfg.get("kappa1", 0.0))}
            ph2={"t_start":t_sw, "direction":"von_mises", "mu": float(cfg.get("mu2", 0.0)), "kappa": float(cfg.get("kappa2", 0.0))}
            return [ph1, ph2]
        if scen=="two_phase_random_switch":
            dist=cfg.get("switch_dist", "lognorm").lower()
            rng=np.random.default_rng(int(self.random.random()*1e9))
            if dist=="lognorm":
                mu=float(cfg.get("switch_meanlog", 4.0)); sd=float(cfg.get("switch_sdlog", 0.6))
                t_sw=float(np.exp(rng.normal(mu, sd)))
            elif dist=="uniform":
                lo=float(cfg.get("switch_low", 10.0)); hi=float(cfg.get("switch_high", 200.0))
                t_sw=float(rng.uniform(lo, hi))
            else:
                t_sw=float(cfg.get("t_switch", 100.0))
            ph1={"t_start":0.0, "direction":"von_mises", "mu": float(cfg.get("mu1", 0.0)), "kappa": float(cfg.get("kappa1", 0.0))}
            ph2={"t_start":t_sw, "direction":"von_mises", "mu": float(cfg.get("mu2", 0.0)), "kappa": float(cfg.get("kappa2", 0.0))}
            return [ph1, ph2]
        return None

    def get_neighbors(self, agent, r):
        if agent.pos is None: return []
        candidates=self.space.get_neighbors(pos=agent.pos, radius=float(r), include_center=False)
        return [a for a in candidates if getattr(a, "alive", True)]

    def remove_agent(self, agent):
        try: self.agents.remove(agent)
        except Exception: pass
        try: self.space.remove_agent(agent)
        except Exception: pass

    def spawn_cluster(self, size, phenotype, pos=None, jitter=False, movement_phases=None):
        if movement_phases is None:
            movement_phases=self._maybe_build_phases_for_new_agent()
        a=ClusterAgent(model=self, size=size, phenotype=phenotype, movement_phases=movement_phases)
        self.agents.add(a)
        if pos is None:
            pos=np.array([self.random.uniform(0,self.params["space"]["width"]), self.random.uniform(0,self.params["space"]["height"])], dtype=float)
        else:
            pos=np.array(pos, dtype=float)
        if jitter:
            jx=self.random.normalvariate(0,1); jy=self.random.normalvariate(0,1)
            pos=pos+np.array([jx,jy], dtype=float)
        x,y=self.space.torus_adj(tuple(pos))
        self.space.place_agent(a,(float(x),float(y)))
        return a

    def _log_alive_snapshot(self):
        ids=[]; xs=[]; ys=[]; radii=[]; sizes=[]; speeds=[]
        for a in list(self.agents):
            if not getattr(a,"alive",True): continue
            p=getattr(a,"pos",None)
            if p is None: continue
            try:
                x=float(p[0]); y=float(p[1]); r=float(a.radius); s=float(a.size); v=float(np.linalg.norm(a.vel)); i=int(a.unique_id)
            except Exception:
                continue
            xs.append(x); ys.append(y); radii.append(r); sizes.append(s); speeds.append(v); ids.append(i)
        if len(xs)==0:
            import numpy as np
            return (np.array([],dtype=int), np.empty((0,2),dtype=float), np.array([],dtype=float), np.array([],dtype=float), np.array([],dtype=float))
        import numpy as np
        pos=np.column_stack([np.array(xs,dtype=float), np.array(ys,dtype=float)])
        return (np.array(ids,dtype=int), pos.astype(float,copy=False), np.array(radii,dtype=float), np.array(sizes,dtype=float), np.array(speeds,dtype=float))

    def step(self):
        self.agents.shuffle_do("step")
        self.time += self.dt
        ids,pos,radii,sizes,speeds=self._log_alive_snapshot()
        self.id_log.append(ids); self.pos_log.append(pos); self.radius_log.append(radii); self.size_log.append(sizes); self.speed_log.append(speeds)

    def _log_state(self):
        ids,pos,radii,sizes,speeds=self._log_alive_snapshot()
        self.id_log.append(ids); self.pos_log.append(pos); self.radius_log.append(radii); self.size_log.append(sizes); self.speed_log.append(speeds)
