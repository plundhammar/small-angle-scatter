from __future__ import annotations

import numpy as np
from .constants import ME_C2_KEV


def kn_unnorm(cos_t: np.ndarray, E_keV: float) -> np.ndarray:
    a = float(E_keV) / ME_C2_KEV
    r = 1.0 / (1.0 + a * (1.0 - cos_t))
    return (r * r) * (r + 1.0 / r - (1.0 - cos_t * cos_t))


def sample_compton_theta(n: int, E_keV: float, rng: np.random.Generator) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, 2048)
    M = float(kn_unnorm(grid, E_keV).max() * 1.05)
    out = np.empty(int(n), dtype=float)
    i = 0
    while i < n:
        m = int(n - i)
        cs = rng.uniform(-1.0, 1.0, size=m)
        u = rng.uniform(0.0, 1.0, size=m)
        keep = (u * M) < kn_unnorm(cs, E_keV)
        k = int(np.sum(keep))
        if k:
            out[i : i + k] = np.arccos(cs[keep])
            i += k
    return out


def compton_energy(E_keV: float, theta_rad: float) -> float:
    a = float(E_keV) / ME_C2_KEV
    return float(E_keV / (1.0 + a * (1.0 - float(np.cos(theta_rad)))))