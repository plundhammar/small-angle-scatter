import numpy as np
import pandas as pd


def load_photon_shards(paths):
    dfs = [pd.read_feather(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def filter_detected_photons(df):
    df = df[df["exited"].to_numpy(bool)]
    df = df[~df["absorbed"].to_numpy(bool)]
    df = df[df["detected"].to_numpy(bool)]
    df = df.dropna(subset=["n_object_scatters", "E_exit_keV"])
    return df.reset_index(drop=True)


def stacked_energy_histograms(E_keV, n_scat, max_scat=5, bins=511):
    E = np.asarray(E_keV, float)
    S = np.asarray(n_scat, int)

    m = np.isfinite(E) & np.isfinite(S)
    E = E[m]
    S = S[m]

    edges = np.linspace(0.0, 511.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = float(edges[1] - edges[0])

    labels = [str(i) for i in range(1, max_scat + 1)]
    labels.append(f">{max_scat}")

    H = []
    for i in range(1, max_scat + 1):
        e = E[S == i]
        e = e[e < 511.0]
        H.append(np.histogram(e, bins=edges)[0])

    e = E[S > max_scat]
    e = e[e < 511.0]
    H.append(np.histogram(e, bins=edges)[0])

    return centers, bw, labels, np.asarray(H)