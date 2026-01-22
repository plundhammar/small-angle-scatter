from __future__ import annotations
import numpy as np
import pandas as pd
from math import pi
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable
from tqdm.auto import tqdm
import os
import pyarrow as pa
import pyarrow.feather as feather


DATA = np.genfromtxt(
    "../data/xcomH2O.tsv",
    delimiter="\t",
    names=True,
)

# Constants
ME_C2_KEV = 511.0
NA        = 6.02214076e23     # 1/mol
RE_CM     = 2.8179403262e-13  # cm

# Geometry & Material
@dataclass
class Cylinder:
    radius_cm: float
    half_len_cm: float  # along x in [-L, +L]

def _loglog_interp_scalar(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """Log-log interpolation for positive x."""
    x = float(x)
    if x <= 0:
        raise ValueError("Energy must be > 0")
    # clamp to tabulated range
    x_clamped = min(max(x, float(xp[0])), float(xp[-1]))
    y = np.exp(np.interp(np.log(x_clamped), np.log(xp), np.log(fp)))
    return float(y)


@dataclass
class WaterLike:
    """Water-like material with energy-dependent attenuation from XCOM table."""
    rho_g_cm3: float = 1.0

    # XCOM tabulated arrays (Energy in keV, coefficients in cm^2/g)
    E_keV: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    coh_cm2_g: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    incoh_cm2_g: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    photo_cm2_g: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    total_w_coh_cm2_g: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    total_wo_coh_cm2_g: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    use_xcom_total: bool = True # If False, total = photo + incoh + coh (if coh is used), else use XCOM total

    # Internal cached interpolators
    _mu_photo_cm1: Optional[Callable[[float], float]] = field(init=False, default=None, repr=False)
    _mu_comp_cm1: Optional[Callable[[float], float]] = field(init=False, default=None, repr=False)
    _mu_coh_cm1: Optional[Callable[[float], float]] = field(init=False, default=None, repr=False)
    _mu_tot_cm1: Optional[Callable[[float], float]] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        # Sort by energy and ensure positivity
        idx = np.argsort(self.E_keV)
        self.E_keV = np.asarray(self.E_keV, float)[idx]

        def _prep(arr: np.ndarray, name: str) -> np.ndarray:
            arr = np.asarray(arr, float)[idx]
            if arr.size and np.any(arr <= 0):
                # Some columns can be zeros (e.g. pair production at low E)
                pass
            return arr

        self.coh_cm2_g = _prep(self.coh_cm2_g, "coh_cm2_g")
        self.incoh_cm2_g = _prep(self.incoh_cm2_g, "incoh_cm2_g")
        self.photo_cm2_g = _prep(self.photo_cm2_g, "photo_cm2_g")
        self.total_w_coh_cm2_g = _prep(self.total_w_coh_cm2_g, "total_w_coh_cm2_g")
        self.total_wo_coh_cm2_g = _prep(self.total_wo_coh_cm2_g, "total_wo_coh_cm2_g")

        # Convert cm^2/g -> 1/cm by multiplying with density (g/cm^3)
        mu_photo = self.photo_cm2_g * self.rho_g_cm3
        mu_comp  = self.incoh_cm2_g * self.rho_g_cm3
        mu_coh   = self.coh_cm2_g * self.rho_g_cm3
        mu_tot   = self.total_w_coh_cm2_g * self.rho_g_cm3 if self.use_xcom_total else (mu_photo + mu_comp + mu_coh)

        # replace any zeros with tiny floor if needed
        tiny = 1e-300
        mu_photo = np.maximum(mu_photo, tiny)
        mu_comp  = np.maximum(mu_comp,  tiny)
        mu_coh   = np.maximum(mu_coh,   tiny)
        mu_tot   = np.maximum(mu_tot,   tiny)

        E = self.E_keV

        self._mu_photo_cm1 = lambda Ek: _loglog_interp_scalar(Ek, E, mu_photo)
        self._mu_comp_cm1  = lambda Ek: _loglog_interp_scalar(Ek, E, mu_comp)
        self._mu_coh_cm1   = lambda Ek: _loglog_interp_scalar(Ek, E, mu_coh)
        self._mu_tot_cm1   = lambda Ek: _loglog_interp_scalar(Ek, E, mu_tot)

    # Public methods
    def mu_compton_cm1(self, E_keV: float) -> float:
        return self._mu_comp_cm1(E_keV)
    
    def mu_photo_cm1(self, E_keV: float) -> float:
        return self._mu_photo_cm1(E_keV)

    def mu_coherent_cm1(self, E_keV: float, use_coherent: bool = False) -> float: # Defaul ignore coherent scattering
        if not use_coherent:
            return 0.0
        return self._mu_coh_cm1(E_keV)

    def mu_total_cm1(self, E_keV: float) -> float:
        return self._mu_tot_cm1(E_keV)

    def mu_components_cm1(self, E_keV: float) -> Tuple[float, float, float, float]:
        muC  = self.mu_compton_cm1(E_keV)
        muPE = self.mu_photo_cm1(E_keV)
        muCO = self.mu_coherent_cm1(E_keV)
        muT  = self.mu_total_cm1(E_keV) if self.use_xcom_total else (muC + muPE + muCO)
        return float(muT), float(muC), float(muPE), float(muCO)

# Klein-Nishina sampling
def _kn_unnorm(cos_t: np.ndarray, E_keV: float) -> np.ndarray:
    a = E_keV / ME_C2_KEV
    r = 1.0 / (1.0 + a*(1.0 - cos_t))
    return (r*r) * (r + 1.0/r - (1.0 - cos_t**2))  # ∝ dσ/dΩ

def _sample_compton_theta(n: int, E_keV: float, rng: np.random.Generator) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, 2048)
    M = float(_kn_unnorm(grid, E_keV).max() * 1.05)
    out = np.empty(n, dtype=float); i = 0
    while i < n:
        m = n - i
        cs = rng.uniform(-1.0, 1.0, size=m)
        u  = rng.uniform(0.0, 1.0, size=m)
        keep = (u * M) < _kn_unnorm(cs, E_keV)
        k = int(keep.sum())
        if k:
            out[i:i+k] = np.arccos(cs[keep]); i += k
    return out

def _compton_energy(E_keV: float, theta_rad: float) -> float:
    a = E_keV / ME_C2_KEV
    return E_keV / (1.0 + a * (1.0 - np.cos(theta_rad)))

# Geometry helpers
def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0.0:
        return None
    return v / n

def _inside_cyl(p: np.ndarray, cyl: Cylinder) -> bool:
    return (p[1]*p[1] + p[2]*p[2] <= cyl.radius_cm**2 + 1e-9) and (abs(p[0]) <= cyl.half_len_cm + 1e-9)

def _dist_to_exit(p: np.ndarray, d: np.ndarray, cyl: Cylinder) -> float:
    # p - point, d - direction (unit), cyl - Cylinder
    R, L = cyl.radius_cm, cyl.half_len_cm

    # End caps
    tx = np.inf
    if abs(d[0]) > 1e-12: # If not parallel to end caps
        tx = (L - p[0]) / d[0] if d[0] > 0 else (-L - p[0]) / d[0] # the distance to end cap (might be outside cylinder mantel)
        if tx < 0: tx = np.inf # behind start point

    # Lateral
    a = d[1]*d[1] + d[2]*d[2] # dy^2 + dz^2
    t_side = np.inf
    if a > 1e-20: # If not parallel to cylinder axis
        b = 2.0*(p[1]*d[1] + p[2]*d[2]) # 2(y0*dy + z0*dz)
        c = p[1]*p[1] + p[2]*p[2] - R*R # y0^2 + z0^2 - R^2
        disc = b*b - 4*a*c # discriminant
        if disc >= 0.0: # If there is an intersection
            sd = np.sqrt(disc)  # sqrt(discriminant)
            t1 = (-b - sd)/(2*a); t2 = (-b + sd)/(2*a) # two solutions
            cand = [t for t in (t1,t2) if t >= 0] # only forward intersections
            if cand: t_side = min(cand) # nearest intersection

    t = min(tx, t_side) # nearest intersection overall, caps or side
    if np.isinf(t): return np.inf   # no intersection

    x = p[0] + t*d[0]; y = p[1] + t*d[1]; z = p[2] + t*d[2] # intersection point
    if (y*y + z*z > R*R + 1e-6) or (abs(x) > L + 1e-6): return np.inf # outside cylinder
    return t

def _rand_iso_dir(rng: np.random.Generator) -> np.ndarray: # isotropic unit vector
    u = rng.uniform(-1.0, 1.0); phi = rng.uniform(0.0, 2*pi); s = np.sqrt(1-u*u)
    return np.array([s*np.cos(phi), s*np.sin(phi), u], float)

def _scatter_dir(d_in: np.ndarray, theta: float, phi: float) -> np.ndarray: # scatter d_in by (theta, phi)
    z = _unit(d_in)
    tmp = np.array([1.0,0.0,0.0]) if abs(z[0]) < 0.9 else np.array([0.0,1.0,0.0])
    x = _unit(np.cross(tmp, z)); y = np.cross(z, x)
    dloc = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return _unit(x*dloc[0] + y*dloc[1] + z*dloc[2])

def _free_path(mu_tot: float, rng: np.random.Generator) -> float:
    return -np.log(max(rng.uniform(), 1e-16)) / mu_tot # exponentially distributed free path

def _angle_deg(u: Optional[np.ndarray], v: Optional[np.ndarray]) -> float:
    if u is None or v is None: return float("nan")
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(c))) 

def _dist_point_to_line(A: np.ndarray, B: np.ndarray, S: np.ndarray) -> float:
    AB = B - A; n = np.linalg.norm(AB)
    if n == 0.0: return float("nan")
    return float(np.linalg.norm(np.cross(A - S, B - S)) / n) 

# Transport
@dataclass
class ExitRec:
    """Photon track termination summary."""
    exited: bool
    absorbed: bool
    pos: Optional[np.ndarray]
    dir: Optional[np.ndarray]
    n_scat: int
    theta_track_deg: float
    E_keV: float

def _transport(
    S: np.ndarray,
    d_init: np.ndarray,
    cyl: Cylinder,
    mat: WaterLike,
    rng: np.random.Generator,
    max_scat: int = 30,
    energy_cut_keV: float = 0.0,
) -> ExitRec:
    """Simulate photon transport through cylinder."""
    if not _inside_cyl(S, cyl):
        raise ValueError("Source outside cylinder")

    p = S.astype(float).copy()
    d = _unit(d_init)
    if d is None:
        d = _unit(_rand_iso_dir(rng))

    E = 511.0
    nsc = 0
    n0 = d.copy()

    while True: # Breaks when photon exits, is absorbed, or has reached max scatters
        muT, muC, muPE, muCO = mat.mu_components_cm1(E)
        if muT <= 0.0:
            return ExitRec(False, True, None, None, nsc, float("nan"), E)

        s_int = _free_path(muT, rng)
        t_exit = _dist_to_exit(p, d, cyl)

        # Photon exits before next interaction
        if t_exit < s_int:
            theta_track = _angle_deg(n0, d)
            return ExitRec(True, False, p + t_exit * d, d.copy(), nsc, theta_track, E)

        # Interaction happens inside
        p = p + s_int * d
        u = rng.uniform()
        pc  = muC / muT
        pp  = muPE / muT
        pco = muCO / muT

        # Compton scattering
        if u < pc:
            th = _sample_compton_theta(1, E, rng)[0]
            ph = rng.uniform(0.0, 2 * np.pi)
            d  = _scatter_dir(d, th, ph)
            E  = _compton_energy(E, th)
            nsc += 1
            if nsc >= max_scat or E <= energy_cut_keV:
                # killed by energy or too many scatters
                return ExitRec(False, True, None, None, nsc, float("nan"), E)
            continue

        # Photoelectric absorption
        if u < pc + pp:
            return ExitRec(False, True, None, None, nsc, float("nan"), E)

        # Coherent elastic (if active)
        if pco > 0.0:
            th = rng.uniform(0.0, np.radians(5.0))
            ph = rng.uniform(0.0, 2 * np.pi)
            d  = _scatter_dir(d, th, ph)
            nsc += 1
            if nsc >= max_scat:
                return ExitRec(False, True, None, None, nsc, float("nan"), E)
            continue

        # fallback — should never hit
        return ExitRec(False, True, None, None, nsc, float("nan"), E)


# Public
def run_MC(
    n_annihilations: int,
    *,
    cylinder_radius_cm: float = 10.0,
    cylinder_half_len_cm: float = 15.0,
    source_pos_cm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rng_seed: Optional[int] = 1234,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate back-to-back 511 keV photons in a water-like cylinder, with energy-dependent μ(E).
    Returns:
      scatter_table: per-photon rows with:
        ['photon_ID','annihilation_ID',
         'n_x','n_y','n_z',
         'first_detected_x','first_detected_y','first_detected_z',
         'theta_eff_deg','cos_theta_eff',
         'theta_track_deg',            
         'n_object_scatters','scattered',
         'exited','absorbed','detected',
         'E_exit_keV']
      lor_tbl: per-annihilation with:
        ['annihilation_ID','photon_ID_1','photon_ID_2',
         'p1_x','p1_y','p1_z','p2_x','p2_y','p2_z',
         'n_scatter_p1','n_scatter_p2',
         'theta_eff_1','theta_eff_2',
         'lor_dist_to_point_cm']
    """
    cyl = Cylinder(cylinder_radius_cm, cylinder_half_len_cm)
    mat = WaterLike(
                        rho_g_cm3=1.0,
                        E_keV=DATA["Energy"] * 1000.0,
                        coh_cm2_g=DATA["Coherent"],
                        incoh_cm2_g=DATA["Incoherent"],
                        photo_cm2_g=DATA["Photoelectric"],
                        total_w_coh_cm2_g=DATA["Total_w_Coherent"],
                        total_wo_coh_cm2_g=DATA["Total_wo_Coherent"],
                        use_xcom_total=True,
                    )
    rng = np.random.default_rng(rng_seed)
    S = np.array(source_pos_cm, float)

    rows = []
    it = tqdm(range(n_annihilations), desc="Simulating annihilations") if show_progress else range(n_annihilations)

    for ann_id in it:
        d0 = _unit(_rand_iso_dir(rng))
        rec1 = _transport(S,  d0, cyl, mat, rng)
        rec2 = _transport(S, -d0, cyl, mat, rng)

        def _det_ok(rec: ExitRec) -> bool:
            if not rec.exited or rec.absorbed:
                return False
            return True
            

        det1 = _det_ok(rec1); det2 = _det_ok(rec2)

        # Build rows
        for j, rec, n_vec in ((1, rec1,  d0), (2, rec2, -d0)):
            photon_id = f"{ann_id}_{j}"

            if rec.pos is not None:
                x, y, z = map(float, rec.pos)
            else:
                x = y = z = np.nan

            # Effective direction: source → exit point
            if rec.pos is None:
                theta_eff = np.nan
                cos_eff = np.nan
            else:
                u_vec = rec.pos - S
                u_hat = _unit(u_vec)
                n_hat = _unit(n_vec)
                cos_eff = np.nan if (u_hat is None or n_hat is None) else float(np.clip(np.dot(n_hat, u_hat), -1.0, 1.0))
                theta_eff = float(np.degrees(np.arccos(cos_eff))) if np.isfinite(cos_eff) else np.nan

            detected = bool(det1) if j == 1 else bool(det2)

            rows.append({
                "photon_ID": photon_id,
                "annihilation_ID": ann_id,
                "n_x": float(n_vec[0]), "n_y": float(n_vec[1]), "n_z": float(n_vec[2]),
                "first_detected_x": float(x),
                "first_detected_y": float(y),
                "first_detected_z": float(z),
                "theta_eff_deg": theta_eff,
                "cos_theta_eff": cos_eff,
                "theta_track_deg": float(rec.theta_track_deg),
                "n_object_scatters": int(rec.n_scat),
                "scattered": bool(rec.n_scat > 0),
                "exited": bool(rec.exited),
                "absorbed": bool(rec.absorbed),
                "detected": detected,
                "E_exit_keV": float(rec.E_keV) if rec.exited else np.nan,
            })

    scatter_table = pd.DataFrame(rows).sort_values(["annihilation_ID","photon_ID"]).reset_index(drop=True)

    # Build LORs
    lor_rows = []
    for ann_id, g in scatter_table.groupby("annihilation_ID", sort=False):
        if len(g) != 2:
            continue
        if not (bool(g.iloc[0]["detected"]) and bool(g.iloc[1]["detected"])):
            continue

        A = np.array([g.iloc[0]["first_detected_x"], g.iloc[0]["first_detected_y"], g.iloc[0]["first_detected_z"]], float)
        B = np.array([g.iloc[1]["first_detected_x"], g.iloc[1]["first_detected_y"], g.iloc[1]["first_detected_z"]], float)
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            continue

        d_pt = _dist_point_to_line(A, B, S)
        AB = B - A
        den = float(np.dot(AB, AB))
        if den == 0.0:
            continue

        t = float(np.dot(S - A, AB) / den)
        P = A + t * AB
        dr = S - P

        lor_rows.append({
            "annihilation_ID": ann_id,
            "photon_ID_1": g.iloc[0]["photon_ID"],
            "photon_ID_2": g.iloc[1]["photon_ID"],
            "p1_x": A[0], "p1_y": A[1], "p1_z": A[2],
            "p2_x": B[0], "p2_y": B[1], "p2_z": B[2],
            "n_scatter_p1": int(g.iloc[0]["n_object_scatters"]),
            "n_scatter_p2": int(g.iloc[1]["n_object_scatters"]),
            "theta_eff_1": float(g.iloc[0]["theta_eff_deg"]),
            "theta_eff_2": float(g.iloc[1]["theta_eff_deg"]),
            "theta_track_1": float(g.iloc[0]["theta_track_deg"]),
            "theta_track_2": float(g.iloc[1]["theta_track_deg"]),
            "lor_dist_to_point_cm": float(d_pt),
            "lor_offset_x_cm": float(dr[0]),
            "lor_offset_y_cm": float(dr[1]),
            "lor_offset_z_cm": float(dr[2]),
            "lor_closest_x": float(P[0]),
            "lor_closest_y": float(P[1]),
            "lor_closest_z": float(P[2]),
            "exit_energy_1_keV": float(g.iloc[0]["E_exit_keV"]),
            "exit_energy_2_keV": float(g.iloc[1]["E_exit_keV"]),
        })

    lor_tbl = pd.DataFrame(lor_rows).sort_values("annihilation_ID").reset_index(drop=True)
    return scatter_table, lor_tbl

def lor_offset_vector(A, B, S):
    AB = B - A
    den = float(np.dot(AB, AB))
    if den == 0.0:
        return np.nan, np.nan, np.nan, np.nan
    t = float(np.dot(S - A, AB) / den)
    P = A + t * AB
    dr = S - P
    d = float(np.linalg.norm(dr))
    return float(dr[0]), float(dr[1]), float(dr[2]), d


def run_MC_fast(
    n_annihilations: int,
    *,
    cylinder_radius_cm: float = 10.0,
    cylinder_half_len_cm: float = 15.0,
    source_pos_cm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rng_seed: Optional[int] = 1234,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Faster MC runner.
    """

    # Setup
    cyl = Cylinder(cylinder_radius_cm, cylinder_half_len_cm)
    mat = WaterLike(
        rho_g_cm3=1.0,
        E_keV=DATA["Energy"] * 1000.0,
        coh_cm2_g=DATA["Coherent"],
        incoh_cm2_g=DATA["Incoherent"],
        photo_cm2_g=DATA["Photoelectric"],
        total_w_coh_cm2_g=DATA["Total_w_Coherent"],
        total_wo_coh_cm2_g=DATA["Total_wo_Coherent"],
        use_xcom_total=True,
    )
    rng = np.random.default_rng(rng_seed)
    S = np.asarray(source_pos_cm, dtype=float)

    # Column-wise buffers for photons
    p_ann = []
    p_idx = []
    n_x = []; n_y = []; n_z = []
    exit_x = []; exit_y = []; exit_z = []
    theta_eff = []; cos_eff = []
    theta_track = []
    n_scat = []
    scattered = []
    exited = []
    absorbed = []
    detected = []
    E_exit = []

    # Buffers for LOR rows
    l_ann = []
    p1_idx = []; p2_idx = []
    p1_x = []; p1_y = []; p1_z = []
    p2_x = []; p2_y = []; p2_z = []
    n_scat_1 = []; n_scat_2 = []
    theta_eff_1 = []; theta_eff_2 = []
    theta_track_1 = []; theta_track_2 = []
    exit_E1 = []; exit_E2 = []
    lor_dist = []

    # Iterator
    if show_progress:
        from tqdm import tqdm
        it = tqdm(range(n_annihilations), desc="Simulating annihilations")
    else:
        it = range(n_annihilations)

    def _theta_eff_deg_and_cos(n_hat: np.ndarray, pos: Optional[np.ndarray]) -> Tuple[float, float]:
        if pos is None:
            return (np.nan, np.nan)
        u = pos - S
        # If u is zero-length, angle undefined
        uu = float(np.dot(u, u))
        if uu == 0.0 or not np.isfinite(uu):
            return (np.nan, np.nan)
        u_hat = u / np.sqrt(uu)
        c = float(np.clip(np.dot(n_hat, u_hat), -1.0, 1.0))
        t = float(np.degrees(np.arccos(c))) if np.isfinite(c) else np.nan
        return (t, c)

    for ann_id in it:
        # d0 is unit by construction
        d0 = _unit(_rand_iso_dir(rng))

        rec1 = _transport(S,  d0, cyl, mat, rng)
        rec2 = _transport(S, -d0, cyl, mat, rng)

        det1 = bool(rec1.exited and (not rec1.absorbed))
        det2 = bool(rec2.exited and (not rec2.absorbed))

        # photon 1
        pos1 = rec1.pos
        if pos1 is None:
            x1 = y1 = z1 = np.nan
        else:
            x1, y1, z1 = float(pos1[0]), float(pos1[1]), float(pos1[2])

        t1, c1 = _theta_eff_deg_and_cos(d0, pos1)

        # Append photon 1 columns
        p_ann.append(ann_id); p_idx.append(1)
        n_x.append(float(d0[0])); n_y.append(float(d0[1])); n_z.append(float(d0[2]))
        exit_x.append(x1); exit_y.append(y1); exit_z.append(z1)
        theta_eff.append(t1); cos_eff.append(c1)
        theta_track.append(float(rec1.theta_track_deg))
        ns1 = int(rec1.n_scat)
        n_scat.append(ns1)
        scattered.append(bool(ns1 > 0))
        exited.append(bool(rec1.exited))
        absorbed.append(bool(rec1.absorbed))
        detected.append(det1)
        E_exit.append(float(rec1.E_keV) if rec1.exited else np.nan)

        # photon 2
        nd2 = -d0  # still unit
        pos2 = rec2.pos
        if pos2 is None:
            x2 = y2 = z2 = np.nan
        else:
            x2, y2, z2 = float(pos2[0]), float(pos2[1]), float(pos2[2])

        t2, c2 = _theta_eff_deg_and_cos(nd2, pos2)

        p_ann.append(ann_id); p_idx.append(2)
        n_x.append(float(nd2[0])); n_y.append(float(nd2[1])); n_z.append(float(nd2[2]))
        exit_x.append(x2); exit_y.append(y2); exit_z.append(z2)
        theta_eff.append(t2); cos_eff.append(c2)
        theta_track.append(float(rec2.theta_track_deg))
        ns2 = int(rec2.n_scat)
        n_scat.append(ns2)
        scattered.append(bool(ns2 > 0))
        exited.append(bool(rec2.exited))
        absorbed.append(bool(rec2.absorbed))
        detected.append(det2)
        E_exit.append(float(rec2.E_keV) if rec2.exited else np.nan)

        # LOR
        if det1 and det2 and (pos1 is not None) and (pos2 is not None):
            A = np.asarray(pos1, dtype=float)
            B = np.asarray(pos2, dtype=float)
            if np.all(np.isfinite(A)) and np.all(np.isfinite(B)):
                d_pt = float(_dist_point_to_line(A, B, S))

                l_ann.append(ann_id)
                p1_idx.append(1); p2_idx.append(2)
                p1_x.append(float(A[0])); p1_y.append(float(A[1])); p1_z.append(float(A[2]))
                p2_x.append(float(B[0])); p2_y.append(float(B[1])); p2_z.append(float(B[2]))
                n_scat_1.append(ns1); n_scat_2.append(ns2)
                theta_eff_1.append(t1); theta_eff_2.append(t2)
                theta_track_1.append(float(rec1.theta_track_deg))
                theta_track_2.append(float(rec2.theta_track_deg))
                exit_E1.append(float(rec1.E_keV))
                exit_E2.append(float(rec2.E_keV))
                lor_dist.append(d_pt)

    photons_df = pd.DataFrame({
        "annihilation_ID": p_ann,
        "photon_index": p_idx,
        "n_x": n_x, "n_y": n_y, "n_z": n_z,
        "exit_x": exit_x, "exit_y": exit_y, "exit_z": exit_z,
        "theta_eff_deg": theta_eff,
        "cos_theta_eff": cos_eff,
        "theta_track_deg": theta_track,
        "n_object_scatters": n_scat,
        "scattered": scattered,
        "exited": exited,
        "absorbed": absorbed,
        "detected": detected,
        "E_exit_keV": E_exit,
    }).sort_values(["annihilation_ID", "photon_index"]).reset_index(drop=True)

    lors_df = pd.DataFrame({
        "annihilation_ID": l_ann,
        "photon_index_1": p1_idx,
        "photon_index_2": p2_idx,
        "p1_x": p1_x, "p1_y": p1_y, "p1_z": p1_z,
        "p2_x": p2_x, "p2_y": p2_y, "p2_z": p2_z,
        "n_scatter_p1": n_scat_1,
        "n_scatter_p2": n_scat_2,
        "theta_eff_1": theta_eff_1,
        "theta_eff_2": theta_eff_2,
        "theta_track_1": theta_track_1,
        "theta_track_2": theta_track_2,
        "exit_energy_1_keV": exit_E1,
        "exit_energy_2_keV": exit_E2,
        "lor_dist_to_point_cm": lor_dist,
    }).sort_values("annihilation_ID").reset_index(drop=True)

    return photons_df, lors_df

#
# Below is the chunked version that writes out to disk directly
#

def write_feather(df, path):
    table = pa.Table.from_pandas(df, preserve_index=False)
    feather.write_feather(table, path, compression="zstd")  

def run_MC_fast_chunked(
    n_annihilations: int,
    *,
    chunk_size: int = 200_000,
    out_dir: str = "mc_out",
    cylinder_radius_cm: float = 10.0,
    cylinder_half_len_cm: float = 15.0,
    source_pos_cm=(0.0, 0.0, 0.0),
    rng_seed: int = 1234,
    show_progress: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    # setup
    cyl = Cylinder(cylinder_radius_cm, cylinder_half_len_cm)
    mat = WaterLike(
        rho_g_cm3=1.0,
        E_keV=DATA["Energy"] * 1000.0,
        coh_cm2_g=DATA["Coherent"],
        incoh_cm2_g=DATA["Incoherent"],
        photo_cm2_g=DATA["Photoelectric"],
        total_w_coh_cm2_g=DATA["Total_w_Coherent"],
        total_wo_coh_cm2_g=DATA["Total_wo_Coherent"],
        use_xcom_total=True,
    )
    rng = np.random.default_rng(rng_seed)
    S = np.asarray(source_pos_cm, dtype=float)

    # progress
    if show_progress:
        from tqdm import tqdm
        it = tqdm(range(0, n_annihilations, chunk_size), desc="Simulating chunks")
    else:
        it = range(0, n_annihilations, chunk_size)

    def _theta_eff_deg_and_cos(n_hat: np.ndarray, pos):
        if pos is None:
            return (np.nan, np.nan)
        u = pos - S
        uu = float(np.dot(u, u))
        if uu == 0.0 or not np.isfinite(uu):
            return (np.nan, np.nan)
        u_hat = u / np.sqrt(uu)
        c = float(np.clip(np.dot(n_hat, u_hat), -1.0, 1.0))
        t = float(np.degrees(np.arccos(c))) if np.isfinite(c) else np.nan
        return (t, c)

    chunk_id = 0

    for start in it:
        stop = min(start + chunk_size, n_annihilations)

        # buffers for THIS chunk
        p_ann=[]; p_idx=[]
        n_x=[]; n_y=[]; n_z=[]
        exit_x=[]; exit_y=[]; exit_z=[]
        theta_eff=[]; cos_eff=[]
        theta_track=[]
        n_scat=[]
        scattered=[]
        exited=[]
        absorbed=[]
        detected=[]
        E_exit=[]

        l_ann=[]
        p1_x=[]; p1_y=[]; p1_z=[]
        p2_x=[]; p2_y=[]; p2_z=[]
        n_scat_1=[]; n_scat_2=[]
        theta_eff_1=[]; theta_eff_2=[]
        theta_track_1=[]; theta_track_2=[]
        exit_E1=[]; exit_E2=[]
        lor_dist=[]
        lor_dx=[]; lor_dy=[]; lor_dz=[]

        for ann_id in range(start, stop):
            d0 = _unit(_rand_iso_dir(rng))
            rec1 = _transport(S,  d0, cyl, mat, rng)
            rec2 = _transport(S, -d0, cyl, mat, rng)

            det1 = bool(rec1.exited and (not rec1.absorbed))
            det2 = bool(rec2.exited and (not rec2.absorbed))

            # photon 1
            pos1 = rec1.pos
            if pos1 is None:
                x1=y1=z1=np.nan
            else:
                x1,y1,z1 = float(pos1[0]), float(pos1[1]), float(pos1[2])
            t1,c1 = _theta_eff_deg_and_cos(d0, pos1)
            ns1 = int(rec1.n_scat)

            p_ann.append(ann_id); p_idx.append(1)
            n_x.append(float(d0[0])); n_y.append(float(d0[1])); n_z.append(float(d0[2]))
            exit_x.append(x1); exit_y.append(y1); exit_z.append(z1)
            theta_eff.append(t1); cos_eff.append(c1)
            theta_track.append(float(rec1.theta_track_deg))
            n_scat.append(ns1); scattered.append(ns1>0)
            exited.append(bool(rec1.exited)); absorbed.append(bool(rec1.absorbed))
            detected.append(det1)
            E_exit.append(float(rec1.E_keV) if rec1.exited else np.nan)

            # photon 2
            nd2 = -d0
            pos2 = rec2.pos
            if pos2 is None:
                x2=y2=z2=np.nan
            else:
                x2,y2,z2 = float(pos2[0]), float(pos2[1]), float(pos2[2])
            t2,c2 = _theta_eff_deg_and_cos(nd2, pos2)
            ns2 = int(rec2.n_scat)

            p_ann.append(ann_id); p_idx.append(2)
            n_x.append(float(nd2[0])); n_y.append(float(nd2[1])); n_z.append(float(nd2[2]))
            exit_x.append(x2); exit_y.append(y2); exit_z.append(z2)
            theta_eff.append(t2); cos_eff.append(c2)
            theta_track.append(float(rec2.theta_track_deg))
            n_scat.append(ns2); scattered.append(ns2>0)
            exited.append(bool(rec2.exited)); absorbed.append(bool(rec2.absorbed))
            detected.append(det2)
            E_exit.append(float(rec2.E_keV) if rec2.exited else np.nan)

            # LOR
            if det1 and det2 and (pos1 is not None) and (pos2 is not None):
                A = np.asarray(pos1, float)
                B = np.asarray(pos2, float)
                if np.all(np.isfinite(A)) and np.all(np.isfinite(B)):
                    dx_, dy_, dz_, dpt = lor_offset_vector(A, B, S)
                    l_ann.append(ann_id)
                    p1_x.append(float(A[0])); p1_y.append(float(A[1])); p1_z.append(float(A[2]))
                    p2_x.append(float(B[0])); p2_y.append(float(B[1])); p2_z.append(float(B[2]))
                    n_scat_1.append(ns1); n_scat_2.append(ns2)
                    theta_eff_1.append(t1); theta_eff_2.append(t2)
                    theta_track_1.append(float(rec1.theta_track_deg))
                    theta_track_2.append(float(rec2.theta_track_deg))
                    exit_E1.append(float(rec1.E_keV))
                    exit_E2.append(float(rec2.E_keV))
                    lor_dx.append(dx_); lor_dy.append(dy_); lor_dz.append(dz_)
                    lor_dist.append(dpt)

        photons_df = pd.DataFrame({
            "annihilation_ID": p_ann,
            "photon_index": p_idx,
            "n_x": n_x, "n_y": n_y, "n_z": n_z,
            "exit_x": exit_x, "exit_y": exit_y, "exit_z": exit_z,
            "theta_eff_deg": theta_eff,
            "cos_theta_eff": cos_eff,
            "theta_track_deg": theta_track,
            "n_object_scatters": n_scat,
            "scattered": scattered,
            "exited": exited,
            "absorbed": absorbed,
            "detected": detected,
            "E_exit_keV": E_exit,
        })

        lors_df = pd.DataFrame({
            "annihilation_ID": l_ann,
            "p1_x": p1_x, "p1_y": p1_y, "p1_z": p1_z,
            "p2_x": p2_x, "p2_y": p2_y, "p2_z": p2_z,
            "n_scatter_p1": n_scat_1,
            "n_scatter_p2": n_scat_2,
            "theta_eff_1": theta_eff_1,
            "theta_eff_2": theta_eff_2,
            "theta_track_1": theta_track_1,
            "theta_track_2": theta_track_2,
            "exit_energy_1_keV": exit_E1,
            "exit_energy_2_keV": exit_E2,
            "lor_offset_x_cm": lor_dx,
            "lor_offset_y_cm": lor_dy,
            "lor_offset_z_cm": lor_dz,
            "lor_dist_to_point_cm": lor_dist,
        })

        # write chunk
        photons_path = os.path.join(out_dir, f"photons_{chunk_id:05d}_R{int(cylinder_radius_cm)}.parquet")
        lors_path    = os.path.join(out_dir, f"lors_{chunk_id:05d}_R{int(cylinder_radius_cm)}.parquet")
        write_feather(photons_df, photons_path.replace(".parquet", ".feather"))
        write_feather(lors_df, lors_path.replace(".parquet", ".feather"))


        chunk_id += 1

    return out_dir
