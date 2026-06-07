from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_pickle(path: str | Path) -> Any:
    """Load a pickle file (pandas first, stdlib fallback)."""
    p = Path(path)
    try:
        return pd.read_pickle(p)
    except Exception:
        with open(p, "rb") as f:
            return pickle.load(f)


def load_trials(path: str | Path) -> dict[str, list[np.ndarray]]:
    """
    Load a pickle and return mic trials with the original key names from the file.

    Returns
    -------
    dict  {key: [1D float array per trial]}
    Keys are whatever the pickle uses (e.g. 'Mic1', 'mic1').
    For ndarray payloads (no keys) the keys are 'mic1', 'mic2', 'mic3'.
    """
    payload = load_pickle(path)

    if isinstance(payload, pd.DataFrame):
        return {
            col: [np.asarray(payload.iloc[i][col], dtype=float).ravel() for i in range(len(payload))]
            for col in payload.columns
        }

    if isinstance(payload, dict):
        return {
            k: [np.asarray(v[i], dtype=float).ravel() for i in range(len(v))]
            for k, v in payload.items()
        }

    arr = np.asarray(payload, dtype=object)

    if arr.ndim == 2 and arr.shape[0] >= 3:  # (n_mics, n_trials), each cell a 1D array
        return {f"mic{i+1}": [np.asarray(arr[i, j], dtype=float).ravel() for j in range(arr.shape[1])] for i in range(3)}

    if arr.ndim == 3 and arr.shape[0] >= 3:  # (n_mics, n_trials, n_samples)
        return {f"mic{i+1}": [np.asarray(arr[i, j, :], dtype=float).ravel() for j in range(arr.shape[1])] for i in range(3)}

    raise ValueError(f"Unsupported pickle payload: type={type(payload)}, shape={getattr(arr, 'shape', None)}")


def store_trials(payload: Any, denoised: dict[str, list[np.ndarray]]) -> Any:
    """
    Write denoised signals back into the original payload structure.

    Parameters
    ----------
    payload  : original pickle payload (DataFrame, dict, or ndarray)
    denoised : same keys as returned by load_trials, each a list of 1D arrays

    Returns
    -------
    Updated payload in the same format as the input.
    """
    if isinstance(payload, pd.DataFrame):
        updated = payload.copy(deep=True)
        for key, trials in denoised.items():
            updated[key] = [
                _coerce(payload.iloc[i][key], trials[i]) for i in range(len(trials))
            ]
        return updated

    if isinstance(payload, dict):
        updated = dict(payload)
        for key, trials in denoised.items():
            updated[key] = [_coerce(payload[key][i], trials[i]) for i in range(len(trials))]
        return updated

    if isinstance(payload, np.ndarray):
        if payload.ndim != 3 or payload.shape[0] < 3:
            raise ValueError(f"Unexpected ndarray shape: {payload.shape}")
        updated = np.array(payload, copy=True)
        for key, trials in denoised.items():
            mic_idx = int(key[-1]) - 1  # 'mic1' -> 0, 'mic2' -> 1, 'mic3' -> 2
            for j, signal in enumerate(trials):
                updated[mic_idx, j] = np.asarray(signal, dtype=float)
        return updated

    raise ValueError(f"Unsupported pickle format: {type(payload)}")


def _coerce(reference: Any, signal: np.ndarray) -> Any:
    """Return signal in the same container type as the original entry."""
    arr = np.asarray(signal, dtype=float)
    if isinstance(reference, tuple):
        return tuple(arr.tolist())
    if isinstance(reference, list):
        return arr.tolist()
    return arr


def iter_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    process_fn,
) -> list[dict]:
    """
    Walk every .pickle in input_dir, call process_fn on each, save the result.

    Parameters
    ----------
    input_dir  : root folder with .pickle files (searched recursively)
    output_dir : root folder to write processed pickles (same subfolder structure)
    process_fn : callable(pickle_path) -> (denoised_dict, summary_rows)
                 denoised_dict  : {key: [1D array per trial]}
                 summary_rows   : list of dicts for the CSV

    Returns
    -------
    All summary rows collected across every file.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    pickle_files = sorted(input_dir.rglob("*.pickle"))
    print(f"Found {len(pickle_files)} pickle files in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for pickle_path in pickle_files:
        rel = pickle_path.relative_to(input_dir)
        out_subdir = output_dir / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        print(f"  {rel} ...", end=" ")
        try:
            denoised, rows = process_fn(pickle_path)
            payload = load_pickle(pickle_path)
            updated = store_trials(payload, denoised)
            with open(out_subdir / pickle_path.name, "wb") as f:
                pickle.dump(updated, f)
            all_rows.extend(rows)
            print(f"OK ({len(rows)} rows)")
        except Exception as exc:
            print(f"ERROR: {exc}")

    return all_rows
