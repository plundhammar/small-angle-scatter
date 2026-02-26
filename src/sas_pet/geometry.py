from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class Cylinder:
    radius_cm: float
    half_len_cm: float



def intersect_cylinder(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    *,
    radius_cm: float,
    half_length_cm: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    x0 = np.asarray(x0, float)
    y0 = np.asarray(y0, float)
    z0 = np.asarray(z0, float)
    nx = np.asarray(nx, float)
    ny = np.asarray(ny, float)
    nz = np.asarray(nz, float)

    a = nx * nx + ny * ny
    b = 2.0 * (x0 * nx + y0 * ny)
    c = x0 * x0 + y0 * y0 - radius_cm * radius_cm

    px = np.full_like(x0, np.nan, dtype=float)
    py = np.full_like(y0, np.nan, dtype=float)
    pz = np.full_like(z0, np.nan, dtype=float)

    ok_dir = a > eps
    disc = b * b - 4.0 * a * c
    ok = ok_dir & (disc >= 0.0)
    if not np.any(ok):
        return px, py, pz

    idx = np.flatnonzero(ok)
    sqrt_disc = np.sqrt(disc[ok])
    a_ok = a[ok]
    b_ok = b[ok]

    t1 = (-b_ok - sqrt_disc) / (2.0 * a_ok)
    t2 = (-b_ok + sqrt_disc) / (2.0 * a_ok)

    z1 = z0[idx] + t1 * nz[idx]
    z2 = z0[idx] + t2 * nz[idx]

    v1 = (t1 > eps) & (np.abs(z1) <= half_length_cm)
    v2 = (t2 > eps) & (np.abs(z2) <= half_length_cm)

    t = np.where(
        v1 & v2,
        np.minimum(t1, t2),
        np.where(v1, t1, np.where(v2, t2, np.nan)),
    )

    keep = np.isfinite(t)
    if not np.any(keep):
        return px, py, pz

    idx2 = idx[keep]
    t_use = t[keep]

    px[idx2] = x0[idx2] + t_use * nx[idx2]
    py[idx2] = y0[idx2] + t_use * ny[idx2]
    pz[idx2] = z0[idx2] + t_use * nz[idx2]

    return px, py, pz

def unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        return None
    return v / n


def inside_cylinder(p: np.ndarray, cyl: Cylinder) -> bool:
    r2 = float(p[1] * p[1] + p[2] * p[2])
    return (r2 <= cyl.radius_cm * cyl.radius_cm + 1e-9) and (abs(float(p[0])) <= cyl.half_len_cm + 1e-9)


def dist_to_exit(p: np.ndarray, d: np.ndarray, cyl: Cylinder) -> float:
    R, L = float(cyl.radius_cm), float(cyl.half_len_cm)

    tx = np.inf
    dx = float(d[0])
    if abs(dx) > 1e-12:
        tx = (L - float(p[0])) / dx if dx > 0 else (-L - float(p[0])) / dx
        if tx < 0:
            tx = np.inf

    a = float(d[1] * d[1] + d[2] * d[2])
    t_side = np.inf
    if a > 1e-20:
        b = float(2.0 * (p[1] * d[1] + p[2] * d[2]))
        c = float(p[1] * p[1] + p[2] * p[2] - R * R)
        disc = float(b * b - 4.0 * a * c)
        if disc >= 0.0:
            sd = float(np.sqrt(disc))
            t1 = float((-b - sd) / (2.0 * a))
            t2 = float((-b + sd) / (2.0 * a))
            cand = [t for t in (t1, t2) if t >= 0.0]
            if cand:
                t_side = min(cand)

    t = min(tx, t_side)
    if not np.isfinite(t):
        return np.inf

    x = float(p[0] + t * d[0])
    y = float(p[1] + t * d[1])
    z = float(p[2] + t * d[2])

    if (y * y + z * z > R * R + 1e-6) or (abs(x) > L + 1e-6):
        return np.inf

    return float(t)


def rand_iso_dir(rng: np.random.Generator) -> np.ndarray:
    u = float(rng.uniform(-1.0, 1.0))
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    s = float(np.sqrt(max(0.0, 1.0 - u * u)))
    return np.array([s * np.cos(phi), s * np.sin(phi), u], float)


def scatter_dir(d_in: np.ndarray, theta: float, phi: float) -> Optional[np.ndarray]:
    z = unit(d_in)
    if z is None:
        return None
    tmp = np.array([1.0, 0.0, 0.0]) if abs(float(z[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = unit(np.cross(tmp, z))
    if x is None:
        return None
    y = np.cross(z, x)
    dloc = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], float)
    return unit(x * dloc[0] + y * dloc[1] + z * dloc[2])


def angle_deg(u: Optional[np.ndarray], v: Optional[np.ndarray]) -> float:
    if u is None or v is None:
        return float("nan")
    c = float(np.clip(float(np.dot(u, v)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def dist_point_to_line(A: np.ndarray, B: np.ndarray, S: np.ndarray) -> float:
    AB = B - A
    n = float(np.linalg.norm(AB))
    if n == 0.0:
        return float("nan")
    return float(np.linalg.norm(np.cross(A - S, B - S)) / n)


def lor_offset_vector(A: np.ndarray, B: np.ndarray, S: np.ndarray) -> Tuple[float, float, float, float]:
    AB = B - A
    den = float(np.dot(AB, AB))
    if den == 0.0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    t = float(np.dot(S - A, AB) / den)
    P = A + t * AB
    dr = S - P
    d = float(np.linalg.norm(dr))
    return float(dr[0]), float(dr[1]), float(dr[2]), float(d)