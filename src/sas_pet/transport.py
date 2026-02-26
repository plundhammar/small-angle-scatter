from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from .geometry import Cylinder, inside_cylinder, unit, rand_iso_dir, dist_to_exit, scatter_dir, angle_deg
from .materials import WaterLike
from .physics import sample_compton_theta, compton_energy


@dataclass
class ExitRec:
    exited: bool
    absorbed: bool
    pos: Optional[np.ndarray]
    dir: Optional[np.ndarray]
    n_scat: int
    theta_track_deg: float
    E_keV: float


def free_path(mu_tot: float, rng: np.random.Generator) -> float:
    u = float(rng.uniform())
    u = max(u, 1e-16)
    return float(-np.log(u) / float(mu_tot))


def transport(
    S: np.ndarray,
    d_init: np.ndarray,
    cyl: Cylinder,
    mat: WaterLike,
    rng: np.random.Generator,
    max_scat: int = 30,
    energy_cut_keV: float = 0.0,
) -> ExitRec:
    if not inside_cylinder(S, cyl):
        raise ValueError("Source outside cylinder")

    p = np.asarray(S, float).copy()
    d = unit(d_init)
    if d is None:
        d = unit(rand_iso_dir(rng))
    if d is None:
        return ExitRec(False, True, None, None, 0, float("nan"), 511.0)

    E = 511.0
    nsc = 0
    n0 = d.copy()

    while True:
        muT, muC, muPE, muCO = mat.mu_components_cm1(E)
        if muT <= 0.0 or not np.isfinite(muT):
            return ExitRec(False, True, None, None, nsc, float("nan"), E)

        s_int = free_path(muT, rng)
        t_exit = dist_to_exit(p, d, cyl)

        if t_exit < s_int:
            return ExitRec(True, False, p + t_exit * d, d.copy(), nsc, angle_deg(n0, d), E)

        p = p + s_int * d

        u = float(rng.uniform())
        pc = float(muC / muT)
        pp = float(muPE / muT)
        pco = float(muCO / muT)

        if u < pc:
            th = float(sample_compton_theta(1, E, rng)[0])
            ph = float(rng.uniform(0.0, 2.0 * np.pi))
            d2 = scatter_dir(d, th, ph)
            if d2 is None:
                return ExitRec(False, True, None, None, nsc, float("nan"), E)
            d = d2
            E = compton_energy(E, th)
            nsc += 1
            if nsc >= max_scat or E <= energy_cut_keV:
                return ExitRec(False, True, None, None, nsc, float("nan"), E)
            continue

        if u < pc + pp:
            return ExitRec(False, True, None, None, nsc, float("nan"), E)

        if pco > 0.0:
            th = float(rng.uniform(0.0, np.radians(5.0)))
            ph = float(rng.uniform(0.0, 2.0 * np.pi))
            d2 = scatter_dir(d, th, ph)
            if d2 is None:
                return ExitRec(False, True, None, None, nsc, float("nan"), E)
            d = d2
            nsc += 1
            if nsc >= max_scat:
                return ExitRec(False, True, None, None, nsc, float("nan"), E)
            continue

        return ExitRec(False, True, None, None, nsc, float("nan"), E)