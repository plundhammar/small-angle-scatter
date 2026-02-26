from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
import numpy as np
import pandas as pd

from sas_pet.geometry import intersect_cylinder

@dataclass(frozen=True)
class DetectorCylinder:
    radius_cm: float
    ring_length_cm: float

    @property
    def z_max_cm(self) -> float:
        return 0.5 * float(self.ring_length_cm)

def accumulate_photon_fates_from_shard(
    path: str | Path,
    *,
    det: DetectorCylinder,
) -> Dict[str, int]:
    need = [
        "photon_index",
        "exit_x", "exit_y", "exit_z",
        "n_x", "n_y", "n_z",
        "n_object_scatters",
        "E_exit_keV",
    ]
    opt = ["exited"]

    df = pd.read_feather(path)
    cols = [c for c in need if c in df.columns] + [c for c in opt if c in df.columns]
    df = df[cols]

    if "photon_index" in df.columns:
        df = df[df["photon_index"].isin([1, 2])]

    if len(df) == 0:
        return dict(
            emitted=0,
            exited=0,
            absorbed=0,
            not_scattered=0,
            scattered=0,
            hit_detector=0,
            missed_detector=0,
        )

    if "exited" in df.columns:
        m_exit = df["exited"].to_numpy(bool)
    else:
        m_exit = np.isfinite(df["E_exit_keV"].to_numpy(float))

    n_emitted = int(len(df))
    n_exited = int(np.sum(m_exit))
    n_absorbed = int(n_emitted - n_exited)

    if n_exited == 0:
        return dict(
            emitted=n_emitted,
            exited=0,
            absorbed=n_absorbed,
            not_scattered=0,
            scattered=0,
            hit_detector=0,
            missed_detector=0,
        )

    dfe = df.loc[m_exit]

    scat = dfe["n_object_scatters"].to_numpy(int)
    n_not_sc = int(np.sum(scat == 0))
    n_sc = int(np.sum(scat > 0))

    px, py, pz = intersect_cylinder(
        dfe["exit_x"].to_numpy(float),
        dfe["exit_y"].to_numpy(float),
        dfe["exit_z"].to_numpy(float),
        dfe["n_x"].to_numpy(float),
        dfe["n_y"].to_numpy(float),
        dfe["n_z"].to_numpy(float),
        radius_cm=float(det.radius_cm),
        half_length_cm=float(det.z_max_cm),
    )
    hit = np.isfinite(px)
    n_hit = int(np.sum(hit))
    n_miss = int(n_exited - n_hit)

    return dict(
        emitted=n_emitted,
        exited=n_exited,
        absorbed=n_absorbed,
        not_scattered=n_not_sc,
        scattered=n_sc,
        hit_detector=n_hit,
        missed_detector=n_miss,
    )

def accumulate_photon_fates(
    shards: Iterable[str | Path],
    *,
    det: DetectorCylinder,
) -> Dict[str, int]:
    tot = dict(
        emitted=0,
        exited=0,
        absorbed=0,
        not_scattered=0,
        scattered=0,
        hit_detector=0,
        missed_detector=0,
    )
    for p in shards:
        inc = accumulate_photon_fates_from_shard(p, det=det)
        for k, v in inc.items():
            tot[k] += int(v)
    return tot

def validate_fates(stats: Dict[str, int]) -> List[str]:
    msgs: List[str] = []
    if stats["exited"] + stats["absorbed"] != stats["emitted"]:
        msgs.append("exited + absorbed != emitted")
    if stats["hit_detector"] + stats["missed_detector"] != stats["exited"]:
        msgs.append("hit_detector + missed_detector != exited")
    if stats["not_scattered"] + stats["scattered"] != stats["exited"]:
        msgs.append("not_scattered + scattered != exited")
    return msgs