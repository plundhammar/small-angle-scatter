from __future__ import annotations

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from pathlib import Path
from typing import List, Tuple
import glob

def write_feather(df: pd.DataFrame, path: str | Path) -> str:
    path = str(path)
    table = pa.Table.from_pandas(df, preserve_index=False)
    feather.write_feather(table, path, compression="zstd")
    return path


def radius_dir(root: str | Path, R: float) -> Path:
    root = Path(root)
    return root / f"R{int(round(float(R)))}"


def photon_shards(root: str | Path, R: float) -> List[Path]:
    d = radius_dir(root, R)
    files = sorted(glob.glob(str(d / "photons_*.feather")))
    return [Path(f) for f in files]


def lor_shards(root: str | Path, R: float) -> List[Path]:
    d = radius_dir(root, R)
    files = sorted(glob.glob(str(d / "lors_*.feather")))
    return [Path(f) for f in files]


def shard_pairs(root: str | Path, R: float) -> Tuple[List[Path], List[Path]]:
    return photon_shards(root, R), lor_shards(root, R)