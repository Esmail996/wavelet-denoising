"""Path-B triangulation runner and library utilities.

How to run
----------
From repository root:
  python -m triangulation.triangulation

With explicit paths:
  python -m triangulation.triangulation \
    --input-csv "outputs/tof_matched/multifrequenz_denoised_branchA_bayes_garrote_calibrated.csv" \
    --output-csv "outputs/triangulation/path_b_calibrated_results.csv"

Useful CLI options
------------------
  --input-csv PATH
      ToF CSV with Path-B columns (tof_path_b_s, amp_path_b) and trial keys.
  --output-csv PATH
      Destination for per-trial triangulation results.
  --no-calibration
      Skip subtraction of empirical Path-B delay constants before solving.
  --max-groups N
      Process only the first N grouped trials (smoke testing).

Function overview
-----------------
  _pair_arrays:
      Converts a {(mic, tx): value} map into ordered arrays for solving.
  _seed_from_pair:
      Builds one geometric start guess from a single Tx/Mic pair and ToF.
  _build_initial_guesses:
      Creates multiple initial guesses for robust multi-start optimization.
  _solve_single_start:
      Runs one damped Gauss-Newton solve and returns estimate + covariance.
  triangulate:
      Core weighted multilateration over bistatic ToFs.
  reflector_angle:
      Estimates reflector normal angle from amplitude pattern consistency.
  build_path_b_inputs:
      Extracts triangulation input maps from Path-B rows, optional calibration.
  triangulate_from_path_b_rows:
      One-call triangulation for a single trial slice DataFrame.
  _truth_xy:
      Converts dataset distance/angle labels to (x,y) ground-truth coordinates.
  run_from_csv:
      Batch runner over all trial groups in an input CSV; writes results CSV.
  _build_arg_parser:
      Defines CLI arguments.
  main:
      CLI entrypoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow direct execution via `python triangulation/triangulation.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOF_DIR = PROJECT_ROOT / "tof"
if str(TOF_DIR) not in sys.path:
    sys.path.insert(0, str(TOF_DIR))

from calibration import DELAY_CALIBRATION_S
from TOF_estimation import PCB, SOUND_SPEED_M_S


def _pair_arrays(values: dict[tuple[str, str], float], geom: dict[str, tuple[float, float]]):
    pairs = list(values.keys())
    if len(pairs) < 2:
        raise ValueError("Need at least 2 (mic, tx) pairs for 2D triangulation.")

    Ts = np.array([geom[tx] for (_, tx) in pairs], dtype=float)
    Ms = np.array([geom[mc] for (mc, _) in pairs], dtype=float)
    ys = np.array([values[k] for k in pairs], dtype=float)
    return pairs, Ts, Ms, ys


def _seed_from_pair(T: np.ndarray, M: np.ndarray, tof_s: float, c: float) -> np.ndarray:
    """Generate a geometric initial guess for one (Tx, Mic) pair."""
    L = float(np.linalg.norm(T - M))
    d0 = float(c) * float(tof_s)
    y0 = 0.5 * np.sqrt(max(d0 * d0 - L * L, 1e-9))
    x0 = 0.5 * (float(T[0]) + float(M[0]))
    return np.array([x0, max(1e-3, y0)], dtype=float)


def _build_initial_guesses(
    Ts: np.ndarray,
    Ms: np.ndarray,
    ts: np.ndarray,
    amps: dict[tuple[str, str], float],
    pairs: list[tuple[str, str]],
    c: float,
) -> list[np.ndarray]:
    """Create diverse initialization candidates for robust convergence."""
    amp_vec = np.array([amps[k] for k in pairs], dtype=float)
    sort_idx = np.argsort(amp_vec)[::-1]

    seeds: list[np.ndarray] = []
    top_k = min(3, len(sort_idx))
    for i in range(top_k):
        idx = int(sort_idx[i])
        seed = _seed_from_pair(Ts[idx], Ms[idx], ts[idx], c)
        seeds.append(seed)

    if len(seeds) == 0:
        seeds.append(np.array([0.0, 0.2], dtype=float))

    # Mirror and lateral variants to escape wrong local basins.
    base = seeds[0]
    seeds.extend(
        [
            np.array([0.0, max(1e-3, base[1])], dtype=float),
            np.array([base[0] + 0.05, max(1e-3, base[1] * 0.8)], dtype=float),
            np.array([base[0] - 0.05, max(1e-3, base[1] * 0.8)], dtype=float),
            np.array([base[0] + 0.10, max(1e-3, base[1] * 1.2)], dtype=float),
            np.array([base[0] - 0.10, max(1e-3, base[1] * 1.2)], dtype=float),
        ]
    )

    # Deduplicate near-identical seeds.
    uniq: list[np.ndarray] = []
    for s in seeds:
        if not any(np.linalg.norm(s - u) < 1e-9 for u in uniq):
            uniq.append(s)
    return uniq


def _solve_single_start(
    R0: np.ndarray,
    Ts: np.ndarray,
    Ms: np.ndarray,
    ts: np.ndarray,
    W: np.ndarray,
    c: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Run one damped Gauss-Newton solve from a fixed start.

    Returns: (R, cov, n_iter, weighted_sse, rms_residual_m)
    """
    R = np.array(R0, dtype=float)
    eps = 1e-12
    n_iter = 0

    for it in range(1, max_iter + 1):
        dT = R - Ts
        dM = R - Ms
        nT = np.clip(np.linalg.norm(dT, axis=1), eps, None)
        nM = np.clip(np.linalg.norm(dM, axis=1), eps, None)

        r = nT + nM - c * ts
        J = dT / nT[:, None] + dM / nM[:, None]

        H = J.T @ W @ J
        g = J.T @ W @ r

        lam = 1e-9 * np.trace(H) if np.isfinite(np.trace(H)) else 1e-9
        H_damped = H + lam * np.eye(2)

        try:
            dR = -np.linalg.solve(H_damped, g)
        except np.linalg.LinAlgError:
            dR = -np.linalg.pinv(H_damped) @ g

        R = R + dR
        n_iter = it
        if np.linalg.norm(dR) < tol:
            break

    dT = R - Ts
    dM = R - Ms
    nT = np.clip(np.linalg.norm(dT, axis=1), eps, None)
    nM = np.clip(np.linalg.norm(dM, axis=1), eps, None)
    r = nT + nM - c * ts
    J = dT / nT[:, None] + dM / nM[:, None]
    H = J.T @ W @ J

    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    weighted_sse = float(r.T @ W @ r)
    rms_residual_m = float(np.sqrt(np.mean(r * r)))
    return R, cov, n_iter, weighted_sse, rms_residual_m


def triangulate(
    tofs: dict[tuple[str, str], float],
    amps: dict[tuple[str, str], float],
    geom: dict[str, tuple[float, float]] = PCB,
    c: float = SOUND_SPEED_M_S,
    max_iter: int = 20,
    tol: float = 1e-6,
    sigma_t_map: dict[tuple[str, str], float] | None = None,
    robust: bool = True,
    min_range_m: float = 0.02,
    max_range_m: float = 3.0,
    max_rms_residual_m: float = 0.25,
):
    """
    Estimate reflector position from bistatic ToFs.

    Args:
        tofs: {(mic, tx): tof_seconds}
        amps: {(mic, tx): amplitude} used for init guess
        geom: coordinate map in metres
        c: sound speed (m/s)
        max_iter: GN iterations
        tol: stop when ||dR|| < tol (m)
        sigma_t_map: optional per-pair timing sigma (seconds)

    Returns:
        R: np.ndarray shape (2,)
        cov: np.ndarray shape (2,2)
        n_iter: int
    """
    if not tofs:
        raise ValueError("tofs is empty.")
    missing = [k for k in tofs if k not in amps]
    if missing:
        raise KeyError(f"amps missing keys: {missing}")

    pairs, Ts, Ms, ts = _pair_arrays(tofs, geom)

    if sigma_t_map is None:
        sigmas_t = np.full(len(pairs), 1e-7, dtype=float)  # 100 ns prior
    else:
        sigmas_t = np.array([float(sigma_t_map.get(k, 1e-7)) for k in pairs], dtype=float)
        sigmas_t = np.clip(sigmas_t, 1e-9, None)

    W = np.diag(1.0 / (c * sigmas_t) ** 2)

    if robust:
        starts = _build_initial_guesses(Ts=Ts, Ms=Ms, ts=ts, amps=amps, pairs=pairs, c=c)
    else:
        s_idx = int(np.argmax([amps[k] for k in pairs]))
        starts = [_seed_from_pair(Ts[s_idx], Ms[s_idx], ts[s_idx], c)]

    best_any: tuple[np.ndarray, np.ndarray, int, float, float] | None = None
    best_valid: tuple[np.ndarray, np.ndarray, int, float, float] | None = None

    for R0 in starts:
        cand = _solve_single_start(R0, Ts, Ms, ts, W, c, max_iter, tol)
        Rc, covc, nitc, costc, rmsc = cand

        if best_any is None or costc < best_any[3]:
            best_any = cand

        dist_c = float(np.linalg.norm(Rc))
        is_valid = (
            np.isfinite(costc)
            and np.isfinite(rmsc)
            and Rc[1] > 0.0
            and min_range_m <= dist_c <= max_range_m
            and rmsc <= max_rms_residual_m
        )
        if is_valid and (best_valid is None or costc < best_valid[3]):
            best_valid = cand

    chosen = best_valid if best_valid is not None else best_any
    if chosen is None:
        raise RuntimeError("Triangulation failed for all initialization attempts.")

    R, cov, n_iter, _, _ = chosen
    return R, cov, n_iter


def reflector_angle(
    R: np.ndarray,
    amps: dict[tuple[str, str], float],
    geom: dict[str, tuple[float, float]] = PCB,
    theta_grid_deg: np.ndarray = np.arange(-45.0, 46.0, 1.0),
) -> float:
    """
    Fit reflector normal angle (degrees) from amplitude pattern consistency.
    """
    if not amps:
        raise ValueError("amps is empty.")

    pairs, Ts, Ms, A_obs = _pair_arrays(amps, geom)
    _ = pairs  # only needed for consistent ordering

    A_obs = np.clip(A_obs, 0.0, None)
    A_obs = A_obs / (A_obs.max() + 1e-12)

    eps = 1e-12
    best_score = -np.inf
    best_th = 0.0

    dT = R - Ts
    dM = Ms - R

    nT = np.clip(np.linalg.norm(dT, axis=1, keepdims=True), eps, None)
    nM = np.clip(np.linalg.norm(dM, axis=1, keepdims=True), eps, None)

    uT = dT / nT
    uM = dM / nM

    # Specular normal aligns with bisector between incident and reflected directions.
    bisect = uT - uM
    bisect_norm = np.clip(np.linalg.norm(bisect, axis=1, keepdims=True), eps, None)
    bisect = bisect / bisect_norm

    for th in np.deg2rad(theta_grid_deg):
        n_hat = np.array([np.sin(th), np.cos(th)], dtype=float)
        cos_dev = np.abs(bisect @ n_hat)  # 1 => ideal specular for that pair
        pred = cos_dev / (cos_dev.max() + 1e-12)
        score = -np.sum((A_obs - pred) ** 2)

        if score > best_score:
            best_score = score
            best_th = th

    return float(np.rad2deg(best_th))


def build_path_b_inputs(
    rows: pd.DataFrame,
    calibrate: bool = True,
    calibration: dict[tuple[str, str, str], float] = DELAY_CALIBRATION_S,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Build {(mic, tx): tof_s} and {(mic, tx): amp} from ToF result rows.

    Expected columns: mic, tx, tof_path_b_s, amp_path_b.
    If calibrate=True, applies per-(Tx,Mic) Path B delay compensation:
    tof_cal = tof_path_b_s - tau(path_b, tx, mic).
    """
    required = {"mic", "tx", "tof_path_b_s", "amp_path_b"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise KeyError(f"Missing required Path B columns: {missing}")

    tofs: dict[tuple[str, str], float] = {}
    amps: dict[tuple[str, str], float] = {}

    for rec in rows.itertuples(index=False):
        mic = str(rec.mic)
        tx = str(rec.tx)
        key = (mic, tx)

        tof_s = float(rec.tof_path_b_s)
        if calibrate:
            tof_s -= float(calibration.get(("path_b", tx, mic), 0.0))

        tofs[key] = tof_s
        amps[key] = float(rec.amp_path_b)

    if len(tofs) < 2:
        raise ValueError("Need at least 2 unique (mic, tx) rows for triangulation.")

    return tofs, amps


def triangulate_from_path_b_rows(
    rows: pd.DataFrame,
    geom: dict[str, tuple[float, float]] = PCB,
    c: float = SOUND_SPEED_M_S,
    calibrate: bool = True,
    calibration: dict[tuple[str, str, str], float] = DELAY_CALIBRATION_S,
    max_iter: int = 20,
    tol: float = 1e-6,
    sigma_t_map: dict[tuple[str, str], float] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Triangulate directly from a Path B trial slice (typically 9 rows).

    Returns:
        R, cov, n_iter, theta_deg
    """
    tofs, amps = build_path_b_inputs(rows=rows, calibrate=calibrate, calibration=calibration)
    R, cov, n_iter = triangulate(
        tofs=tofs,
        amps=amps,
        geom=geom,
        c=c,
        max_iter=max_iter,
        tol=tol,
        sigma_t_map=sigma_t_map,
    )
    theta_deg = reflector_angle(R, amps=amps, geom=geom)
    return R, cov, n_iter, theta_deg


DEFAULT_INPUT_CSV = Path("outputs") / "tof_matched" / "multifrequenz_denoised_branchA_bayes_garrote_calibrated.csv"
DEFAULT_OUTPUT_CSV = Path("outputs") / "triangulation" / "path_b_calibrated_results.csv"


def _truth_xy(distance_cm: float, angle_deg: float) -> tuple[float, float]:
    d_m = float(distance_cm) * 1e-2
    th = np.deg2rad(float(angle_deg))
    return float(d_m * np.sin(th)), float(d_m * np.cos(th))


def run_from_csv(
    input_csv: Path,
    output_csv: Path,
    calibrate: bool = True,
    max_groups: int | None = None,
) -> pd.DataFrame:
    if not input_csv.exists():
        suggestions = sorted(input_csv.parent.glob("*_calibrated.csv")) if input_csv.parent.exists() else []
        hint = ""
        if suggestions:
            hint = "\nAvailable calibrated files:\n  " + "\n  ".join(str(p) for p in suggestions[:8])
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n"
            "Pass --input-csv with a ToF calibrated file from the TOF pipeline."
            f"{hint}"
        )

    df = pd.read_csv(input_csv)
    required = {
        "category",
        "file",
        "distance_cm",
        "angle_deg",
        "trial",
        "mic",
        "tx",
        "tof_path_b_s",
        "amp_path_b",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    rows_out: list[dict] = []
    group_cols = ["category", "file", "distance_cm", "angle_deg", "trial"]
    groups = df.groupby(group_cols, sort=True, dropna=False)

    for idx, (group_key, g) in enumerate(groups, start=1):
        if max_groups is not None and idx > int(max_groups):
            break

        category, file_name, distance_cm, angle_deg, trial = group_key
        x_true_m, y_true_m = _truth_xy(float(distance_cm), float(angle_deg))

        rec = {
            "category": str(category),
            "file": str(file_name),
            "distance_cm": float(distance_cm),
            "angle_deg": float(angle_deg),
            "trial": int(trial),
            "x_true_m": x_true_m,
            "y_true_m": y_true_m,
            "ok": False,
            "x_est_m": np.nan,
            "y_est_m": np.nan,
            "theta_est_deg": np.nan,
            "xy_err_cm": np.nan,
            "dist_err_cm": np.nan,
            "angle_err_deg": np.nan,
            "n_iter": np.nan,
            "cov_xx": np.nan,
            "cov_xy": np.nan,
            "cov_yy": np.nan,
            "error": "",
        }

        try:
            R, cov, n_iter, theta_est_deg = triangulate_from_path_b_rows(
                rows=g,
                calibrate=calibrate,
            )
            x_est_m = float(R[0])
            y_est_m = float(R[1])

            est_dist_cm = 100.0 * float(np.hypot(x_est_m, y_est_m))
            xy_err_cm = 100.0 * float(np.hypot(x_est_m - x_true_m, y_est_m - y_true_m))
            dist_err_cm = est_dist_cm - float(distance_cm)
            angle_err_deg = float(theta_est_deg) - float(angle_deg)

            rec.update(
                {
                    "ok": True,
                    "x_est_m": x_est_m,
                    "y_est_m": y_est_m,
                    "theta_est_deg": float(theta_est_deg),
                    "xy_err_cm": xy_err_cm,
                    "dist_err_cm": dist_err_cm,
                    "angle_err_deg": angle_err_deg,
                    "n_iter": int(n_iter),
                    "cov_xx": float(cov[0, 0]),
                    "cov_xy": float(cov[0, 1]),
                    "cov_yy": float(cov[1, 1]),
                }
            )
        except Exception as exc:
            rec["error"] = str(exc)

        rows_out.append(rec)

    out_df = pd.DataFrame(rows_out)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Path B triangulation from TOF CSV and save per-trial results."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Do not subtract path_b empirical delay constants before triangulation.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional limit on number of trial groups to process.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    out_df = run_from_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        calibrate=not args.no_calibration,
        max_groups=args.max_groups,
    )

    n_ok = int((out_df["ok"] == True).sum()) if not out_df.empty else 0
    print(f"Processed {len(out_df)} trial groups")
    print(f"Successful: {n_ok}")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()