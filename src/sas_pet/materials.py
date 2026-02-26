from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
import numpy as np
from .xcom import XcomTable, loglog_interp_scalar


@dataclass
class WaterLike:
    rho_g_cm3: float = 1.0
    xcom: Optional[XcomTable] = None
    use_xcom_total: bool = True
    include_coherent: bool = False

    _mu_photo_cm1: Optional[Callable[[float], float]] = field(init=False, default=None, repr=False)
    _mu_comp_cm1: Optional[Callable[[float], float]] = field(init=False, default=None, repr=False)
    _mu_coh_cm1: Optional[Callable[[float], float]] = field(init=False, default=None, repr=False)
    _mu_tot_cm1: Optional[Callable[[float], float]] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.xcom is None:
            raise ValueError("xcom must be provided")

        E = np.asarray(self.xcom.E_keV, float)

        mu_photo = np.asarray(self.xcom.photoelectric_cm2_g, float) * self.rho_g_cm3
        mu_comp = np.asarray(self.xcom.incoherent_cm2_g, float) * self.rho_g_cm3
        mu_coh = np.asarray(self.xcom.coherent_cm2_g, float) * self.rho_g_cm3

        if self.use_xcom_total:
            mu_tot = np.asarray(self.xcom.total_w_coherent_cm2_g, float) * self.rho_g_cm3
        else:
            mu_tot = mu_photo + mu_comp + (mu_coh if self.include_coherent else 0.0)

        tiny = 1e-300
        mu_photo = np.maximum(mu_photo, tiny)
        mu_comp = np.maximum(mu_comp, tiny)
        mu_coh = np.maximum(mu_coh, tiny)
        mu_tot = np.maximum(mu_tot, tiny)

        self._mu_photo_cm1 = lambda Ek: loglog_interp_scalar(Ek, E, mu_photo)
        self._mu_comp_cm1 = lambda Ek: loglog_interp_scalar(Ek, E, mu_comp)
        self._mu_coh_cm1 = lambda Ek: loglog_interp_scalar(Ek, E, mu_coh)
        self._mu_tot_cm1 = lambda Ek: loglog_interp_scalar(Ek, E, mu_tot)

    def mu_photo_cm1(self, E_keV: float) -> float:
        return float(self._mu_photo_cm1(E_keV))

    def mu_compton_cm1(self, E_keV: float) -> float:
        return float(self._mu_comp_cm1(E_keV))

    def mu_coherent_cm1(self, E_keV: float) -> float:
        if not self.include_coherent:
            return 0.0
        return float(self._mu_coh_cm1(E_keV))

    def mu_total_cm1(self, E_keV: float) -> float:
        return float(self._mu_tot_cm1(E_keV))

    def mu_components_cm1(self, E_keV: float) -> Tuple[float, float, float, float]:
        muC = self.mu_compton_cm1(E_keV)
        muPE = self.mu_photo_cm1(E_keV)
        muCO = self.mu_coherent_cm1(E_keV)
        muT = self.mu_total_cm1(E_keV) if self.use_xcom_total else (muC + muPE + muCO)
        return float(muT), float(muC), float(muPE), float(muCO)