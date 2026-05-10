import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MIC_MAP = {"mic1": 0, "mic2": 1, "mic3": 2}


def iter_pickle_files(data_root: Path):
    for category_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        for fp in sorted(category_dir.glob("*.pickle")):
            yield category_dir.name, fp


def load_mic_trials(obj, mic_name: str):
    mic_lower = mic_name.lower()

    if isinstance(obj, pd.DataFrame):
        col_lut = {c.lower(): c for c in obj.columns}
        if mic_lower not in col_lut:
            raise ValueError(
                f"Expected column '{mic_name}' in DataFrame. Got columns: {list(obj.columns)}"
            )
        col = col_lut[mic_lower]
        return [np.asarray(obj.iloc[i][col], dtype=float) for i in range(len(obj))]

    arr = np.asarray(obj, dtype=object)
    if arr.ndim != 2 or arr.shape[0] < 3:
        raise ValueError(f"Unsupported pickle array shape: {arr.shape}")

    mic_idx = MIC_MAP[mic_lower]
    if mic_idx >= arr.shape[0]:
        raise ValueError(f"Array does not contain {mic_name}. Shape: {arr.shape}")

    return [np.asarray(arr[mic_idx, j], dtype=float) for j in range(arr.shape[1])]


def mean_groups(trials, group_size: int):
    if len(trials) < group_size:
        raise ValueError(f"Not enough trials: {len(trials)} < group_size={group_size}")

    n_groups = len(trials) // group_size
    if n_groups == 0:
        raise ValueError("No groups can be formed.")

    out = []
    for g in range(n_groups):
        i0 = g * group_size
        i1 = i0 + group_size
        chunk = trials[i0:i1]

        lengths = {len(x) for x in chunk}
        if len(lengths) != 1:
            raise ValueError(
                f"Signals in one group have different lengths: {sorted(lengths)}"
            )

        stack = np.stack(chunk, axis=0)
        out.append(np.mean(stack, axis=0))
    return out


def save_group_file(out_path: Path, mic_signals: dict[str, np.ndarray]):
    row = {"Mic1": mic_signals["mic1"], "Mic2": mic_signals["mic2"], "Mic3": mic_signals["mic3"]}
    df = pd.DataFrame([row])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)


def process_file(category: str, file_path: Path, out_root: Path, group_size: int):
    obj = pd.read_pickle(file_path)

    mic_group_means = {}
    for mic in ["mic1", "mic2", "mic3"]:
        trials = load_mic_trials(obj, mic)
        mic_group_means[mic] = mean_groups(trials, group_size=group_size)

    n_groups = len(mic_group_means["mic1"])
    for mic in ["mic2", "mic3"]:
        if len(mic_group_means[mic]) != n_groups:
            raise ValueError("Different number of groups across mics.")

    stem = file_path.stem
    for g in range(n_groups):
        trial_start = g * group_size
        trial_end = trial_start + group_size - 1
        out_name = f"{stem}__mean_trials_{trial_start:03d}_{trial_end:03d}.pickle"
        out_path = out_root / category / out_name

        save_group_file(
            out_path,
            {
                "mic1": mic_group_means["mic1"][g],
                "mic2": mic_group_means["mic2"][g],
                "mic3": mic_group_means["mic3"][g],
            },
        )

    return n_groups


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Average trial groups for Mic1/Mic2/Mic3 in each pickle file. "
            "With 100 trials and group_size=10, each source file creates 10 averaged files."
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="Multifrequenz Dataset/Multifrequenz",
        help="Root folder containing category subfolders with .pickle files.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="Multifrequenz Dataset/Multifrequenz_mean10",
        help="Output root folder.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=10,
        help="Number of trials per mean group.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    total_in_files = 0
    total_out_files = 0

    for category, fp in iter_pickle_files(data_root):
        total_in_files += 1
        n_groups = process_file(
            category=category,
            file_path=fp,
            out_root=out_root,
            group_size=int(args.group_size),
        )
        total_out_files += n_groups

    print(f"Input files processed: {total_in_files}")
    print(f"Output averaged files created: {total_out_files}")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()
