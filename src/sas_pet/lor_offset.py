from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .utils import photon_shards
from .geometry import intersect_cylinder


@dataclass(frozen=True)
class DetectorRingConfig:
    R_det_cm: float
    ring_length_cm: float

    @property
    def z_max_cm(self) -> float:
        return 0.5 * float(self.ring_length_cm)


@dataclass(frozen=True)
class OffsetHistConfig:
    bin_width_r_cm: float
    bin_width_z_cm: float
    rperp_max_by_R: Dict[float, float]
    zabs_max_by_R: Dict[float, float]

    @property
    def bins_r(self) -> int:
        return int(max(self.rperp_max_by_R.values()) / float(self.bin_width_r_cm)) + 1

    @property
    def bins_z(self) -> int:
        return int(max(self.zabs_max_by_R.values()) / float(self.bin_width_z_cm)) + 1


def _accumulate_hist_from_photon_file(
    feather_path: Path,
    *,
    det: DetectorRingConfig,
    rperp_max: float,
    zabs_max: float,
    bins_r: int,
    bins_z: int,
    energy_thresholds_keV: np.ndarray,
    counts_r_accum: np.ndarray,
    counts_z_accum: np.ndarray,
) -> None:
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
        return

    df = df[keep].dropna()
    if len(df) == 0:
        return

    df = df[df["photon_index"].isin([1, 2])]
    if len(df) == 0:
        return

    if "exited" in df.columns:
        df = df[df["exited"].to_numpy(bool)]
    if "absorbed" in df.columns:
        df = df[~df["absorbed"].to_numpy(bool)]
    if "detected" in df.columns:
        df = df[df["detected"].to_numpy(bool)]
    if len(df) == 0:
        return

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
        return

    df = df.sort_values(["annihilation_ID", "photon_index"], kind="mergesort")
    vc = df["annihilation_ID"].value_counts(sort=False)
    good_ids = vc.index[vc.to_numpy() == 2]
    df = df[df["annihilation_ID"].isin(good_ids)]
    if len(df) == 0:
        return

    df = df.sort_values(["annihilation_ID", "photon_index"], kind="mergesort")

    P = df[["px", "py", "pz"]].to_numpy(float).reshape(-1, 2, 3)
    S = df[["n_object_scatters"]].to_numpy(int).reshape(-1, 2)
    E = df[["E_exit_keV"]].to_numpy(float).reshape(-1, 2)

    m_scatter = (S[:, 0] > 0) | (S[:, 1] > 0)
    if not np.any(m_scatter):
        return

    P = P[m_scatter]
    E = E[m_scatter]

    p1 = P[:, 0, :]
    p2 = P[:, 1, :]

    v = p2 - p1
    vv = np.sum(v * v, axis=1)
    ok = vv > 1e-12
    if not np.any(ok):
        return

    p1 = p1[ok]
    p2 = p2[ok]
    E = E[ok]

    v = p2 - p1
    vv = np.sum(v * v, axis=1)
    s = -np.sum(p1 * v, axis=1) / vv
    closest = p1 + s[:, None] * v

    r_perp = np.sqrt(closest[:, 0] ** 2 + closest[:, 1] ** 2)
    z_abs = np.abs(closest[:, 2])

    m = np.isfinite(r_perp) & np.isfinite(z_abs) & (r_perp >= 0.0) & (z_abs >= 0.0) & (r_perp <= rperp_max) & (z_abs <= zabs_max)
    if not np.any(m):
        return

    r_perp = r_perp[m]
    z_abs = z_abs[m]
    E = E[m]
    Emin_pair = np.minimum(E[:, 0], E[:, 1])

    for j, Eth in enumerate(energy_thresholds_keV):
        mE = Emin_pair >= Eth
        if not np.any(mE):
            continue
        hr, _ = np.histogram(r_perp[mE], bins=bins_r, range=(0.0, rperp_max))
        hz, _ = np.histogram(z_abs[mE], bins=bins_z, range=(0.0, zabs_max))
        counts_r_accum[j] += hr
        counts_z_accum[j] += hz


def compute_lor_offset_histograms(
    root: str | Path,
    Rs: Iterable[float],
    *,
    det: DetectorRingConfig,
    hist: OffsetHistConfig,
    energy_thresholds_keV: np.ndarray,
) -> Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    root = Path(root)
    energy_thresholds_keV = np.asarray(energy_thresholds_keV, float)

    bins_r = int(hist.bins_r)
    bins_z = int(hist.bins_z)

    hist_r_by_R: Dict[float, np.ndarray] = {}
    hist_z_by_R: Dict[float, np.ndarray] = {}

    for R_obj in Rs:
        files = photon_shards(root, float(R_obj))
        if not files:
            raise FileNotFoundError(f"No photon shards found for R={float(R_obj)} in {root}")

        rmax = float(hist.rperp_max_by_R[float(R_obj)])
        zmax = float(hist.zabs_max_by_R[float(R_obj)])

        counts_r = np.zeros((len(energy_thresholds_keV), bins_r), dtype=np.int64)
        counts_z = np.zeros((len(energy_thresholds_keV), bins_z), dtype=np.int64)

        for f in files:
            _accumulate_hist_from_photon_file(
                f,
                det=det,
                rperp_max=rmax,
                zabs_max=zmax,
                bins_r=bins_r,
                bins_z=bins_z,
                energy_thresholds_keV=energy_thresholds_keV,
                counts_r_accum=counts_r,
                counts_z_accum=counts_z,
            )

        hist_r_by_R[float(R_obj)] = counts_r
        hist_z_by_R[float(R_obj)] = counts_z

    return hist_r_by_R, hist_z_by_R