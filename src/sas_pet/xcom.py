from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


def loglog_interp_scalar(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    x = float(x)
    if x <= 0.0:
        raise ValueError("x must be > 0")
    x0 = float(xp[0])
    x1 = float(xp[-1])
    xc = min(max(x, x0), x1)
    y = np.exp(np.interp(np.log(xc), np.log(xp), np.log(fp)))
    return float(y)


@dataclass(frozen=True)
class XcomTable:
    E_keV: np.ndarray
    coherent_cm2_g: np.ndarray
    incoherent_cm2_g: np.ndarray
    photoelectric_cm2_g: np.ndarray
    total_w_coherent_cm2_g: np.ndarray
    total_wo_coherent_cm2_g: np.ndarray


def read_xcom_tsv(path: str | Path) -> XcomTable:
    arr = np.genfromtxt(str(path), delimiter="\t", names=True)
    E_keV = np.asarray(arr["Energy"], float) * 1000.0
    idx = np.argsort(E_keV)

    def col(name: str) -> np.ndarray:
        return np.asarray(arr[name], float)[idx]

    return XcomTable(
        E_keV=np.asarray(E_keV[idx], float),
        coherent_cm2_g=col("Coherent"),
        incoherent_cm2_g=col("Incoherent"),
        photoelectric_cm2_g=col("Photoelectric"),
        total_w_coherent_cm2_g=col("Total_w_Coherent"),
        total_wo_coherent_cm2_g=col("Total_wo_Coherent"),
    )