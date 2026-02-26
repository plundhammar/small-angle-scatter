from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .geometry import Cylinder, unit, rand_iso_dir, dist_point_to_line, lor_offset_vector
from .materials import WaterLike
from .transport import transport
from .xcom import read_xcom_tsv
from .utils import write_feather


@dataclass(frozen=True)
class SimulationConfig:
    cylinder_radius_cm: float = 10.0
    cylinder_half_len_cm: float = 15.0
    source_pos_cm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rng_seed: Optional[int] = 1234
    show_progress: bool = True


def run_annihilations(
    n_annihilations: int,
    *,
    cfg: SimulationConfig,
    xcom_path: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cyl = Cylinder(cfg.cylinder_radius_cm, cfg.cylinder_half_len_cm)
    xcom = read_xcom_tsv(xcom_path)
    mat = WaterLike(rho_g_cm3=1.0, xcom=xcom, use_xcom_total=True, include_coherent=False)

    rng = np.random.default_rng(cfg.rng_seed)
    S = np.asarray(cfg.source_pos_cm, float)

    rows = []
    it = tqdm(range(int(n_annihilations)), desc="Simulating annihilations") if cfg.show_progress else range(int(n_annihilations))

    for ann_id in it:
        d0 = unit(rand_iso_dir(rng))
        if d0 is None:
            continue

        rec1 = transport(S, d0, cyl, mat, rng)
        rec2 = transport(S, -d0, cyl, mat, rng)

        det1 = bool(rec1.exited and (not rec1.absorbed))
        det2 = bool(rec2.exited and (not rec2.absorbed))

        for j, rec, n_vec, det in ((1, rec1, d0, det1), (2, rec2, -d0, det2)):
            photon_id = f"{ann_id}_{j}"
            if rec.pos is None:
                x = y = z = np.nan
                theta_eff = np.nan
                cos_eff = np.nan
            else:
                x, y, z = float(rec.pos[0]), float(rec.pos[1]), float(rec.pos[2])
                u_vec = rec.pos - S
                uu = float(np.dot(u_vec, u_vec))
                if uu == 0.0 or not np.isfinite(uu):
                    theta_eff = np.nan
                    cos_eff = np.nan
                else:
                    u_hat = u_vec / np.sqrt(uu)
                    cos_eff = float(np.clip(float(np.dot(unit(n_vec), u_hat)), -1.0, 1.0))
                    theta_eff = float(np.degrees(np.arccos(cos_eff))) if np.isfinite(cos_eff) else np.nan

            rows.append(
                {
                    "photon_ID": photon_id,
                    "annihilation_ID": int(ann_id),
                    "photon_index": int(j),
                    "n_x": float(n_vec[0]),
                    "n_y": float(n_vec[1]),
                    "n_z": float(n_vec[2]),
                    "exit_x": float(x),
                    "exit_y": float(y),
                    "exit_z": float(z),
                    "theta_eff_deg": float(theta_eff),
                    "cos_theta_eff": float(cos_eff),
                    "theta_track_deg": float(rec.theta_track_deg),
                    "n_object_scatters": int(rec.n_scat),
                    "scattered": bool(rec.n_scat > 0),
                    "exited": bool(rec.exited),
                    "absorbed": bool(rec.absorbed),
                    "detected": bool(det),
                    "E_exit_keV": float(rec.E_keV) if rec.exited else np.nan,
                }
            )

    photons = pd.DataFrame(rows).sort_values(["annihilation_ID", "photon_index"]).reset_index(drop=True)

    lor_rows = []
    for ann_id, g in photons.groupby("annihilation_ID", sort=False):
        if len(g) != 2:
            continue
        if not (bool(g.iloc[0]["detected"]) and bool(g.iloc[1]["detected"])):
            continue

        A = np.array([g.iloc[0]["exit_x"], g.iloc[0]["exit_y"], g.iloc[0]["exit_z"]], float)
        B = np.array([g.iloc[1]["exit_x"], g.iloc[1]["exit_y"], g.iloc[1]["exit_z"]], float)
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            continue

        d_pt = float(dist_point_to_line(A, B, S))
        dx, dy, dz, _ = lor_offset_vector(A, B, S)

        lor_rows.append(
            {
                "annihilation_ID": int(ann_id),
                "p1_x": float(A[0]),
                "p1_y": float(A[1]),
                "p1_z": float(A[2]),
                "p2_x": float(B[0]),
                "p2_y": float(B[1]),
                "p2_z": float(B[2]),
                "n_scatter_p1": int(g.iloc[0]["n_object_scatters"]),
                "n_scatter_p2": int(g.iloc[1]["n_object_scatters"]),
                "theta_eff_1": float(g.iloc[0]["theta_eff_deg"]),
                "theta_eff_2": float(g.iloc[1]["theta_eff_deg"]),
                "theta_track_1": float(g.iloc[0]["theta_track_deg"]),
                "theta_track_2": float(g.iloc[1]["theta_track_deg"]),
                "exit_energy_1_keV": float(g.iloc[0]["E_exit_keV"]),
                "exit_energy_2_keV": float(g.iloc[1]["E_exit_keV"]),
                "lor_dist_to_point_cm": float(d_pt),
                "lor_offset_x_cm": float(dx),
                "lor_offset_y_cm": float(dy),
                "lor_offset_z_cm": float(dz),
            }
        )

    lors = pd.DataFrame(lor_rows).sort_values("annihilation_ID").reset_index(drop=True)
    return photons, lors


def run_chunked(
    n_annihilations: int,
    *,
    chunk_size: int,
    out_dir: str | Path,
    cfg: SimulationConfig,
    xcom_path: str | Path,
) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_indices = range(0, int(n_annihilations), int(chunk_size))
    it = tqdm(list(start_indices), desc="Simulating chunks") if cfg.show_progress else start_indices

    chunk_id = 0
    for start in it:
        stop = min(int(n_annihilations), start + int(chunk_size))
        cfg2 = SimulationConfig(
            cylinder_radius_cm=cfg.cylinder_radius_cm,
            cylinder_half_len_cm=cfg.cylinder_half_len_cm,
            source_pos_cm=cfg.source_pos_cm,
            rng_seed=(cfg.rng_seed or 0) + int(start),
            show_progress=False,
        )
        photons, lors = run_annihilations(stop - start, cfg=cfg2, xcom_path=xcom_path)
        p_path = out_dir / f"photons_{chunk_id:05d}_R{int(cfg.cylinder_radius_cm)}.feather"
        l_path = out_dir / f"lors_{chunk_id:05d}_R{int(cfg.cylinder_radius_cm)}.feather"
        write_feather(photons, p_path)
        write_feather(lors, l_path)
        chunk_id += 1

    return str(out_dir)