from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NAME_RE = re.compile(r"(?P<dist>\d+)\s*cm[_-](?P<ang>-?\d+)\s*Grad", re.IGNORECASE)


def _parse_distance_angle(name: str) -> tuple[int | None, int | None]:
    m = NAME_RE.search(name)
    if not m:
        return None, None
    return int(m.group("dist")), int(m.group("ang"))


def _extract_trials_from_payload(payload: Any) -> dict[str, list[np.ndarray]]:
    """
    Normalize multiple pickle payload formats into lower-case mic trial lists.

    Returns
    -------
    dict with keys: mic1, mic2, mic3
        Each value is a list of 1D numpy arrays (one array per trial).
    """
    if isinstance(payload, pd.DataFrame):
        col_lut = {c.lower(): c for c in payload.columns}
        needed = ["mic1", "mic2", "mic3"]
        missing = [m for m in needed if m not in col_lut]
        if missing:
            raise ValueError(
                f"DataFrame is missing expected mic columns {missing}. "
                f"Available columns: {list(payload.columns)}"
            )

        out: dict[str, list[np.ndarray]] = {}
        for mic in needed:
            col = col_lut[mic]
            out[mic] = [np.asarray(payload.iloc[i][col], dtype=float).ravel() for i in range(len(payload))]
        return out

    if isinstance(payload, dict):
        key_lut = {str(k).lower(): k for k in payload.keys()}
        needed = ["mic1", "mic2", "mic3"]
        missing = [m for m in needed if m not in key_lut]
        if missing:
            raise ValueError(
                f"Dict payload is missing expected mic keys {missing}. "
                f"Available keys: {list(payload.keys())}"
            )

        out = {}
        for mic in needed:
            arr = payload[key_lut[mic]]
            out[mic] = [np.asarray(arr[i], dtype=float).ravel() for i in range(len(arr))]
        return out

    arr = np.asarray(payload, dtype=object)

    # Common layout: (n_mics, n_trials) with each entry a 1D array.
    if arr.ndim == 2 and arr.shape[0] >= 3:
        return {
            "mic1": [np.asarray(arr[0, j], dtype=float).ravel() for j in range(arr.shape[1])],
            "mic2": [np.asarray(arr[1, j], dtype=float).ravel() for j in range(arr.shape[1])],
            "mic3": [np.asarray(arr[2, j], dtype=float).ravel() for j in range(arr.shape[1])],
        }

    # Numeric layout: (n_mics, n_trials, n_samples).
    if arr.ndim == 3 and arr.shape[0] >= 3:
        return {
            "mic1": [np.asarray(arr[0, j, :], dtype=float).ravel() for j in range(arr.shape[1])],
            "mic2": [np.asarray(arr[1, j, :], dtype=float).ravel() for j in range(arr.shape[1])],
            "mic3": [np.asarray(arr[2, j, :], dtype=float).ravel() for j in range(arr.shape[1])],
        }

    raise ValueError(f"Unsupported pickle payload type/shape: type={type(payload)}, shape={getattr(arr, 'shape', None)}")


def load_data_folder(folder: str | Path) -> pd.DataFrame:
    """
    Load all .pickle files in one folder into a trial-wise dataframe.

    Output columns
    --------------
    - file
    - distance_cm
    - angle_deg
    - meas_idx
    - mic1
    - mic2
    - mic3
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    rows: list[dict[str, Any]] = []
    for fp in sorted(folder.glob("*.pickle")):
        payload = pd.read_pickle(fp)
        trials = _extract_trials_from_payload(payload)

        n_trials = len(trials["mic1"])
        if not (len(trials["mic2"]) == n_trials and len(trials["mic3"]) == n_trials):
            raise ValueError(f"Mic trial count mismatch in {fp}")

        dist_cm, angle_deg = _parse_distance_angle(fp.name)

        for i in range(n_trials):
            rows.append(
                {
                    "file": fp.name,
                    "distance_cm": dist_cm,
                    "angle_deg": angle_deg,
                    "meas_idx": int(i),
                    "mic1": trials["mic1"][i],
                    "mic2": trials["mic2"][i],
                    "mic3": trials["mic3"][i],
                }
            )

    return pd.DataFrame(rows)
