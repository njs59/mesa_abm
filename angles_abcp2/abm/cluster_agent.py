from mesa import Agent
import numpy as np
from .utils import (radius_from_size_3d, volume_conserving_radius, mass_from_size, momentum_merge)

class ClusterAgent(Agent):
    def __init__(self, model, size, phenotype, movement_phases=None):
        super().__init__(model)
        self.size=int(size)
        self.phenotype=phenotype
        self.radius=radius_from_size_3d(self.size)
        self.vel=np.zeros(2,dtype=float)
        self.alive=True
        self.event_log=[]
        self.movement_phases=None
        if movement_phases is None:
            phases = self.model.params.get("movement_phases", None)
        else:
            phases = movement_phases
        if phases:
            def _tstart(ph): return float(ph.get("t_start",0.0))
            self.movement_phases = sorted([dict(p) for p in phases], key=_tstart)
        self._theta=None

    def step(self):
        self._move()
        self._try_merge()
        self._maybe_proliferate()
        self._maybe_fragment()

    def _active_motion_spec(self):
        mv=self.model.params.get("movement", {})
        spec={
            "speed_mode": mv.get("mode","constant"),
            "speed_value": None,
            "speed_distribution": mv.get("distribution","lognorm"),
            "speed_dist_params": mv.get("dist_params",{}),
            "direction": mv.get("direction","isotropic"),
            "heading_sigma": float(mv.get("heading_sigma",0.25)),
            "mu": float(mv.get("mu",0.0)),
            "kappa": float(mv.get("kappa",2.0)),
        }
        if self.movement_phases:
            t=float(self.model.time)
            active=None
            for ph in self.movement_phases:
                if float(ph.get("t_start",0.0)) <= t:
                    active=ph
                else:
                    break
            if active is None:
                active=self.movement_phases[0]
            for k in ["speed_mode","speed_value","speed_distribution","speed_dist_params","direction","heading_sigma","mu","kappa"]:
                if k in active: spec[k]=active[k]
        return spec

    def _move(self):
        if self.pos is None: return
        ph=self.model.params["phenotypes"][self.phenotype]
        speed_base=float(ph["speed_base"])
        beta=float(ph.get("speed_size_exp",0.0))
        if beta!=0.0:
            speed_base = speed_base * (self.size ** (-beta))
        dt=float(self.model.dt)
        spec=self._active_motion_spec()
        if spec["speed_mode"]=="constant":
            speed = float(spec["speed_value"]) if (spec["speed_value"] is not None) else speed_base
            step_mag=speed*dt
        else:
            dist_name=str(spec["speed_distribution"]).lower(); dp=spec["speed_dist_params"] or {}; eps=1e-12
            if dist_name=="lognorm":
                s=float(dp.get("s",0.6)); scale=float(dp.get("scale",2.0)); z=self.model.random.normalvariate(0.0,s); step_mag=max(scale*np.exp(z),eps)
            elif dist_name=="gamma":
                a=float(dp.get("a",2.0)); sc=float(dp.get("scale",1.0)); step_mag=max(np.random.default_rng().gamma(shape=a, scale=sc), eps)
            elif dist_name=="weibull":
                c=float(dp.get("c",1.5)); sc=float(dp.get("scale",2.0)); U=self.model.random.random(); step_mag=max(sc*(-np.log(max(1.0-U,eps)))**(1.0/c), eps)
            elif dist_name=="rayleigh":
                sc=float(dp.get("scale",2.0)); U=self.model.random.random(); step_mag=max(sc*np.sqrt(-2.0*np.log(max(1.0-U,eps))), eps)
            elif dist_name=="expon":
                sc=float(dp.get("scale",1.0)); U=self.model.random.random(); step_mag=max(-sc*np.log(max(1.0-U,eps)), eps)
            elif dist_name=="invgauss":
                mu=float(dp.get("mu",1.0)); sc=float(dp.get("scale",1.0)); step_mag=max(np.random.default_rng().wald(mean=mu, scale=sc), eps)
            else:
                step_mag=speed_base*dt
        pos_curr=np.asarray(self.pos,dtype=float)
        direction_mode=str(spec["direction"]).lower()
        if direction_mode=="isotropic":
            theta=self.model.random.uniform(-np.pi, np.pi)
        elif direction_mode=="persistent":
            theta_prev=self._theta if (self._theta is not None) else self.model.random.uniform(-np.pi, np.pi)
            sigma=float(spec.get("heading_sigma",0.25))
            theta=float(theta_prev + self.model.random.normalvariate(0.0, sigma))
            self._theta=theta
        elif direction_mode=="von_mises":
            mu=float(spec.get("mu",0.0)); kappa=float(spec.get("kappa",2.0))
            theta=float(np.random.default_rng().vonmises(mu=mu, kappa=max(kappa,0.0)))
            self._theta=theta
        else:
            theta=self.model.random.uniform(-np.pi, np.pi)
        dir_vec=np.array([np.cos(theta), np.sin(theta)], dtype=float)
        self.vel=(step_mag/max(dt,1e-12))*dir_vec
        new_pos=pos_curr + dir_vec*step_mag
        x,y=float(new_pos[0]), float(new_pos[1])
        self.model.space.move_agent(self,(x,y))
        params_physics=self.model.params.get("physics",{})
        if params_physics.get("soft_separate", True):
            r_self=float(self.radius)
            neighs=self.model.get_neighbors(self, r=2.4*r_self)
            pos_self=np.asarray(self.pos,dtype=float)
            disp=np.zeros(2,dtype=float)
            softness=float(params_physics.get("softness",0.15))
            for other in neighs:
                if (other is self) or (other.pos is None) or (not other.alive): continue
                rij=pos_self - np.asarray(other.pos,dtype=float)
                d=float(np.linalg.norm(rij)); r_sum=r_self + float(other.radius)
                if d < r_sum:
                    if d < 1e-12:
                        phi=self.model.random.uniform(-np.pi,np.pi); rij_unit=np.array([np.cos(phi),np.sin(phi)],dtype=float)
                    else:
                        rij_unit=rij/d
                    delta=(r_sum-d)*softness
                    disp += delta*rij_unit
            if np.any(disp):
                newp=pos_self+disp
                self.model.space.move_agent(self,(float(newp[0]), float(newp[1])))

    def _try_merge(self):
        if self.pos is None: return
        merge_cfg=self.model.params["merge"]
        p_merge=float(merge_cfg.get("p_merge",0.9)); p_merge=min(max(p_merge,0.0),1.0)
        neighbors=self.model.get_neighbors(self, 2*self.radius)
        pos_self=np.asarray(self.pos,dtype=float)
        contacts=[]
        for other in neighbors:
            if other is self or not other.alive or other.pos is None: continue
            pos_other=np.asarray(other.pos,dtype=float)
            d2=np.sum((pos_other-pos_self)**2)
            r_sum=self.radius + other.radius
            if d2 <= r_sum*r_sum:
                contacts.append((d2, other))
        if not contacts: return
        d2_min=min(d2 for d2,_ in contacts); tol=1e-12
        tied=[other for d2,other in contacts if abs(d2-d2_min)<=tol]
        target=self.model.random.choice(tied)
        if self.model.random.random() < p_merge:
            self._merge_with(target)

    def _merge_with(self, other):
        p_self=np.array(self.pos,dtype=float); p_other=np.array(other.pos,dtype=float)
        m1=mass_from_size(self.size); m2=mass_from_size(other.size)
        size_new=self.size+other.size
        r_new=volume_conserving_radius(self.radius, other.radius)
        v_new=momentum_merge(m1, self.vel, m2, other.vel)
        pos_new=(m1*p_self + m2*p_other)/(m1+m2)
        other.alive=False
        self.model.remove_agent(other)
        self.size=int(size_new); self.radius=float(r_new); self.vel=v_new
        x,y=float(pos_new[0]), float(pos_new[1])
        self.model.space.move_agent(self,(x,y))
        self.event_log.append(("merge", other.unique_id, self.model.time))

    def _maybe_proliferate(self):
        ph=self.model.params["phenotypes"][self.phenotype]
        lam=ph["prolif_rate"]*self.size*self.model.dt
        if self.model.random.random() < lam:
            self.size += 1
            self.radius = radius_from_size_3d(self.size)
            self.event_log.append(("proliferate", 1, self.model.time))

    def _maybe_fragment(self):
        ph=self.model.params["phenotypes"][self.phenotype]
        lam=ph["fragment_rate"]*self.model.dt
        if self.size>1 and self.model.random.random() < lam:
            self.size -= 1
            self.radius = radius_from_size_3d(self.size)
            if self.pos is None: return
            p_self=np.array(self.pos,dtype=float)
            from .utils import radius_from_size_3d as rsz
            r_child=float(rsz(1))
            factor=float(self.model.params.get("physics",{}).get("fragment_minsep_factor",1.1))
            min_sep=factor*(self.radius + r_child)
            theta=self.model.random.uniform(-np.pi,np.pi)
            offset=min_sep*np.array([np.cos(theta), np.sin(theta)], dtype=float)
            candidate_pos=p_self + offset
            child_pos=self.model.space.torus_adj(tuple(candidate_pos))
            child=self.model.spawn_cluster(size=1, phenotype=self.phenotype, pos=child_pos, jitter=False)
            self.event_log.append(("fragment", child.unique_id, self.model.time))
