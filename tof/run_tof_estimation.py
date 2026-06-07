from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Allow direct execution via `python tof/run_tof_estimation.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TOF_estimation import GATE_CM, FS_HZ, PCB, tof_path_A, tof_path_B
from helpers.data_rw import load_trials


NAME_RE = re.compile(r"(?P<dist>\d+)\s*cm[_-](?P<ang>-?\d+)\s*Grad", re.IGNORECASE)
MICS = ("Mic1", "Mic2", "Mic3")
TXS = ("Tx1", "Tx5", "Tx8")
DEFAULT_INPUT_DIR = Path("Multifrequenz Dataset") / "Multifrequenz_denoised_branchA_sym6"
DEFAULT_OUTPUT_CSV = Path("outputs") / "tof_matched" / "multifrequenz_denoised_branchA_sym6_peaks.csv"


def parse_dist_angle(filename: str) -> tuple[int | None, int | None]:
    match = NAME_RE.search(filename)
    if not match:
        return None, None
    return int(match.group("dist")), int(match.group("ang"))


def run_all(
    data_root: Path,
    out_csv: Path,
    geom: dict[str, tuple[float, float]] = PCB,
    fs_hz: float = FS_HZ,
    gate_cm: float = GATE_CM,
    picker: str = "max",
    max_files: int | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    processed_files = 0
    pickle_files = sorted(data_root.rglob("*.pickle"))

    for pickle_path in pickle_files:
        rel_path = pickle_path.relative_to(data_root).as_posix()
        category = rel_path.split("/", 1)[0] if "/" in rel_path else pickle_path.parent.name
        dist_cm, angle_deg = parse_dist_angle(pickle_path.name)
        if dist_cm is None:
            print(f"Skipping {pickle_path.name}: no distance/angle pattern found.")
            continue

        processed_files += 1
        if max_files is not None and processed_files > max_files:
            break

        trial_map = load_trials(pickle_path)
        key_lut = {key.lower(): key for key in trial_map}
        mic_trials = {}
        for mic_name in MICS:
            lookup_key = mic_name.lower()
            if lookup_key not in key_lut:
                raise KeyError(f"Missing channel '{mic_name}' in loaded keys {list(trial_map)}.")
            mic_trials[mic_name] = trial_map[key_lut[lookup_key]]

        distance_m = float(dist_cm) * 1e-2

        print(f"Processing {rel_path} ...")
        for mic_name, trials in mic_trials.items():
            for trial_idx, signal in enumerate(trials):
                for tx_name in TXS:
                    # Search interval comes from the per-distance STFT window
                    # (selected internally by distance_m); gate_cm is only the
                    # fallback half-width for distances with no measured window.
                    tof_a_s, amp_a, idx_a = tof_path_A(
                        signal,
                        distance_m=distance_m,
                        mic_name=mic_name,
                        tx_name=tx_name,
                        geom=geom,
                        fs_hz=fs_hz,
                        fallback_gate_cm=gate_cm,
                        picker=picker,
                        use_bandpass=True,
                    )
                    tof_a_nobp_s, amp_a_nobp, idx_a_nobp = tof_path_A(
                        signal,
                        distance_m=distance_m,
                        mic_name=mic_name,
                        tx_name=tx_name,
                        geom=geom,
                        fs_hz=fs_hz,
                        fallback_gate_cm=gate_cm,
                        picker=picker,
                        use_bandpass=False,
                    )
                    tof_b_s, amp_b, idx_b = tof_path_B(
                        signal,
                        distance_m=distance_m,
                        mic_name=mic_name,
                        tx_name=tx_name,
                        geom=geom,
                        fs_hz=fs_hz,
                        fallback_gate_cm=gate_cm,
                        picker=picker,
                    )
                    rows.append(
                        {
                            "category": category,
                            "file": pickle_path.name,
                            "relative_path": rel_path,
                            "distance_cm": int(dist_cm),
                            "angle_deg": int(angle_deg),
                            "trial": int(trial_idx),
                            "mic": mic_name,
                            "tx": tx_name,
                            "tof_path_a_s": float(tof_a_s),
                            "tof_path_a_us": float(tof_a_s * 1e6),
                            "amp_path_a": float(amp_a),
                            "peak_idx_path_a": int(idx_a),
                            "tof_path_a_nobp_s": float(tof_a_nobp_s),
                            "tof_path_a_nobp_us": float(tof_a_nobp_s * 1e6),
                            "amp_path_a_nobp": float(amp_a_nobp),
                            "peak_idx_path_a_nobp": int(idx_a_nobp),
                            "tof_path_b_s": float(tof_b_s),
                            "tof_path_b_us": float(tof_b_s * 1e6),
                            "amp_path_b": float(amp_b),
                            "peak_idx_path_b": int(idx_b),
                        }
                    )

    result_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_csv, index=False)
    return result_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ToF Path A and Path B over the denoised dataset and save peak/ToF-only CSV results."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_INPUT_DIR, help="Root folder containing category subfolders with .pickle files.")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Destination CSV path for ToF results.")
    parser.add_argument("--fs-hz", type=float, default=FS_HZ, help="Sampling rate in Hz.")
    parser.add_argument("--gate-cm", type=float, default=GATE_CM, help="Fallback gate half-width in cm, used only for distances with no measured STFT window.")
    parser.add_argument("--picker", type=str, default="max", choices=("max", "nearest"), help="Peak picker inside the window: 'max' (default) or legacy 'nearest'.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for quick smoke tests.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    result_df = run_all(
        data_root=args.data_root,
        out_csv=args.out_csv,
        fs_hz=float(args.fs_hz),
        gate_cm=float(args.gate_cm),
        picker=str(args.picker),
        max_files=args.max_files,
    )
    print(f"Wrote {len(result_df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()


"""
python -m tof.run_tof_estimation `
  --data-root "Multifrequenz Dataset/Multifrequenz_denoised_branchA_bayes_garrote" `
  --out-csv "outputs/tof_matched/multifrequenz_denoised_branchA_bayes_garrote_peaks.csv" `
  --picker max
  
"""
