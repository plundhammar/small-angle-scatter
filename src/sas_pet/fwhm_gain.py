from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .geometry import intersect_cylinder
from .utils import photon_shards


@dataclass(frozen=True)
class DetectorRingConfig:
    R_det_cm: float
    ring_length_cm: float

    @property
    def z_max_cm(self) -> float:
        return 0.5 * float(self.ring_length_cm)


@dataclass(frozen=True)
class FwhmGainConfig:
    edges_r: np.ndarray
    edges_z: np.ndarray
    energy_thresholds_keV: np.ndarray
    n_boot: int = 500
    q_low: float = 0.005
    q_high: float = 0.995
    rng_seed: int = 123
    min_n: int = 30


def hwhm_from_counts(counts: np.ndarray, edges: np.ndarray) -> float:
    counts = np.asarray(counts, float)
    if counts.size < 3:
        return float("nan")
    centers = 0.5 * (edges[:-1] + edges[1:])
    i_peak = int(np.argmax(counts))
    cmax = float(counts[i_peak])
    if not np.isfinite(cmax) or cmax <= 0.0:
        return float("nan")
    half = 0.5 * cmax
    post = counts[i_peak:]
    below = np.where(post <= half)[0]
    if below.size == 0:
        return float("nan")
    i2 = i_peak + int(below[0])
    if i2 <= 0:
        return float("nan")
    x1 = float(centers[i2 - 1])
    x2 = float(centers[i2])
    y1 = float(counts[i2 - 1])
    y2 = float(counts[i2])
    if y2 == y1:
        return x2
    return float(x1 + (half - y1) * (x2 - x1) / (y2 - y1))


def hwhm_bootstrap_ci_from_counts(
    counts: np.ndarray,
    edges: np.ndarray,
    *,
    n_boot: int,
    q_low: float,
    q_high: float,
    rng_seed: int,
    min_n: int,
) -> Tuple[float, Tuple[float, float]]:
    counts = np.asarray(counts, int)
    n = int(counts.sum())
    if n < int(min_n):
        return float("nan"), (float("nan"), float("nan"))
    p = counts / float(n)
    rng = np.random.default_rng(int(rng_seed))
    vals = np.empty(int(n_boot), dtype=float)
    for k in range(int(n_boot)):
        boot = rng.multinomial(n, p)
        vals[k] = hwhm_from_counts(boot, edges)
    vals = vals[np.isfinite(vals)]
    if vals.size < max(10, int(0.33 * int(n_boot))):
        return float("nan"), (float("nan"), float("nan"))
    h_mean = float(np.mean(vals))
    lo = float(np.quantile(vals, float(q_low)))
    hi = float(np.quantile(vals, float(q_high)))
    return h_mean, (lo, hi)


def _accumulate_shard(
    feather_path: Path,
    *,
    det: DetectorRingConfig,
    edges_r: np.ndarray,
    edges_z: np.ndarray,
    energy_thresholds_keV: np.ndarray,
    hist_r_extra: np.ndarray,
    hist_z_extra: np.ndarray,
) -> int:
    df = pd.read_feather(feather_path)

    need = [
        "annihilation_ID", "photon_index",
        "exit_x", "exit_y", "exit_z",
        "n_x", "n_y", "n_z",
        "n_object_scatters",
        "E_exit_keV",
        "exited", "absorbed", "detected",
    ]
    keep = [c for c in need if c in df.columns]
    if not keep:
        return 0

    df = df[keep].dropna()
    if len(df) == 0:
        return 0

    df = df[df["photon_index"].isin([1, 2])]
    if len(df) == 0:
        return 0

    if "exited" in df.columns:
        df = df[df["exited"].to_numpy(bool)]
    if "absorbed" in df.columns:
        df = df[~df["absorbed"].to_numpy(bool)]
    if "detected" in df.columns:
        df = df[df["detected"].to_numpy(bool)]
    if len(df) == 0:
        return 0

    px, py, pz = intersect_cylinder(
        df["exit_x"].to_numpy(float),
        df["exit_y"].to_numpy(float),
        df["exit_z"].to_numpy(float),
        df["n_x"].to_numpy(float),
        df["n_y"].to_numpy(float),
        df["n_z"].to_numpy(float),
        radius_cm=float(det.R_det_cm),
        half_length_cm=float(det.z_max_cm),
    )

    df = df.assign(px=px, py=py, pz=pz).dropna(subset=["px", "py", "pz"])
    if len(df) == 0:
        return 0

    df = df.sort_values(["annihilation_ID", "photon_index"], kind="mergesort")
    vc = df["annihilation_ID"].value_counts(sort=False)
    good_ids = vc.index[vc.to_numpy() == 2]
    df = df[df["annihilation_ID"].isin(good_ids)]
    if len(df) == 0:
        return 0

    df = df.sort_values(["annihilation_ID", "photon_index"], kind="mergesort")

    P = df[["px", "py", "pz"]].to_numpy(float).reshape(-1, 2, 3)
    S = df[["n_object_scatters"]].to_numpy(int).reshape(-1, 2)
    E = df[["E_exit_keV"]].to_numpy(float).reshape(-1, 2)

    p1 = P[:, 0, :]
    p2 = P[:, 1, :]

    v = p2 - p1
    vv = np.sum(v * v, axis=1)
    ok = vv > 1e-12
    if not np.any(ok):
        return 0

    p1 = p1[ok]
    p2 = p2[ok]
    S = S[ok]
    E = E[ok]

    v = p2 - p1
    vv = np.sum(v * v, axis=1)
    s = -np.sum(p1 * v, axis=1) / vv
    closest = p1 + s[:, None] * v

    r_perp = np.sqrt(closest[:, 0] ** 2 + closest[:, 1] ** 2)
    z_abs = np.abs(closest[:, 2])

    m = np.isfinite(r_perp) & np.isfinite(z_abs) & (r_perp >= 0.0) & (z_abs >= 0.0) & (r_perp <= edges_r[-1]) & (z_abs <= edges_z[-1])
    if not np.any(m):
        return 0

    r_perp = r_perp[m]
    z_abs = z_abs[m]
    S = S[m]
    E = E[m]

    baseline = (S[:, 0] == 0) & (S[:, 1] == 0)
    baseline_count = int(np.sum(baseline))

    extra = (S[:, 0] > 0) | (S[:, 1] > 0)
    if not np.any(extra):
        return baseline_count

    r_extra = r_perp[extra]
    z_extra = z_abs[extra]
    E_extra = E[extra]
    Emin = np.minimum(E_extra[:, 0], E_extra[:, 1])

    for j, Eth in enumerate(energy_thresholds_keV):
        mE = Emin >= Eth
        if not np.any(mE):
            continue
        hr, _ = np.histogram(r_extra[mE], bins=edges_r)
        hz, _ = np.histogram(z_extra[mE], bins=edges_z)
        hist_r_extra[j] += hr
        hist_z_extra[j] += hz

    return baseline_count


def compute_fwhm_gain_table_for_R(
    root: str | Path,
    R_obj: float,
    *,
    det: DetectorRingConfig,
    cfg: FwhmGainConfig,
) -> pd.DataFrame:
    root = Path(root)
    files = photon_shards(root, float(R_obj))
    if not files:
        raise FileNotFoundError(f"No photon shards found for R={float(R_obj)} in {root}")

    energy_thresholds_keV = np.asarray(cfg.energy_thresholds_keV, float)
    edges_r = np.asarray(cfg.edges_r, float)
    edges_z = np.asarray(cfg.edges_z, float)

    nbins_r = len(edges_r) - 1
    nbins_z = len(edges_z) - 1

    hist_r_extra = np.zeros((len(energy_thresholds_keV), nbins_r), dtype=np.int64)
    hist_z_extra = np.zeros((len(energy_thresholds_keV), nbins_z), dtype=np.int64)

    baseline_total = 0
    for f in files:
        baseline_total += _accumulate_shard(
            f,
            det=det,
            edges_r=edges_r,
            edges_z=edges_z,
            energy_thresholds_keV=energy_thresholds_keV,
            hist_r_extra=hist_r_extra,
            hist_z_extra=hist_z_extra,
        )

    rows = []
    for j, Eth in enumerate(energy_thresholds_keV):
        cr = hist_r_extra[j]
        cz = hist_z_extra[j]
        N_extra = int(cr.sum())
        N0 = int(baseline_total)
        gain_pct = (float(N_extra) / float(N0) * 100.0) if N0 > 0 else float("nan")

        H_perp, (H_perp_lo, H_perp_hi) = hwhm_bootstrap_ci_from_counts(
            cr, edges_r,
            n_boot=int(cfg.n_boot),
            q_low=float(cfg.q_low),
            q_high=float(cfg.q_high),
            rng_seed=int(cfg.rng_seed),
            min_n=int(cfg.min_n),
        )
        F_perp = 2.0 * float(H_perp)
        F_perp_lo = 2.0 * float(H_perp_lo)
        F_perp_hi = 2.0 * float(H_perp_hi)

        H_z, (H_z_lo, H_z_hi) = hwhm_bootstrap_ci_from_counts(
            cz, edges_z,
            n_boot=int(cfg.n_boot),
            q_low=float(cfg.q_low),
            q_high=float(cfg.q_high),
            rng_seed=int(cfg.rng_seed),
            min_n=int(cfg.min_n),
        )
        F_z = 2.0 * float(H_z)
        F_z_lo = 2.0 * float(H_z_lo)
        F_z_hi = 2.0 * float(H_z_hi)

        rows.append({
            "R_cm": float(R_obj),
            "Eth_keV": float(Eth),
            "N_unscattered": int(N0),
            "N_extra": int(N_extra),
            "Gain_pct": float(gain_pct),
            "FWHM_perp_cm": float(F_perp),
            "FWHM_perp_lo_cm": float(F_perp_lo),
            "FWHM_perp_hi_cm": float(F_perp_hi),
            "FWHM_z_cm": float(F_z),
            "FWHM_z_lo_cm": float(F_z_lo),
            "FWHM_z_hi_cm": float(F_z_hi),
        })

    return pd.DataFrame(rows).sort_values("Eth_keV", ascending=False).reset_index(drop=True)


def compute_fwhm_gain_tables(
    root: str | Path,
    Rs: Iterable[float],
    *,
    det: DetectorRingConfig,
    cfg: FwhmGainConfig,
) -> Dict[float, pd.DataFrame]:
    out: Dict[float, pd.DataFrame] = {}
    for R in Rs:
        out[float(R)] = compute_fwhm_gain_table_for_R(root, float(R), det=det, cfg=cfg)
    return out