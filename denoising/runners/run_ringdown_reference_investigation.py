from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this file directly via: python denoising/runners/....py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from denoising.ringdown_handeling import ringdown_energy_db_reduction, ringdown_handle
from helpers.data_rw import load_trials


DIST_RE = re.compile(r"(?P<cm>\d+)cm", flags=re.IGNORECASE)


def _parse_distance_m(filename: str) -> float | None:
    m = DIST_RE.search(filename)
    if m is None:
        return None
    return float(m.group("cm")) / 100.0


def _parse_int_csv(text: str) -> list[int]:
    vals = [v.strip() for v in text.split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected at least one integer value in CSV list.")
    out = sorted({int(v) for v in vals})
    return out


def _parse_optional_distance_cm_filter(text: str | None) -> set[int] | None:
    if text is None:
        return None
    return set(_parse_int_csv(text))


def _build_ref_templates(ref_pickle: Path) -> dict[str, np.ndarray]:
    ref_trials = load_trials(ref_pickle)
    templates: dict[str, np.ndarray] = {}
    for mic, trials in ref_trials.items():
        stack = np.stack([np.asarray(x, dtype=float) for x in trials], axis=0)
        templates[mic] = np.mean(stack, axis=0)
    return templates


def _build_configs(
    optimize: bool,
    align_search: int,
    corr_end_samples: int,
    fit_end_samples: int,
    align_search_grid: str,
    corr_end_grid: str,
    fit_end_grid: str,
) -> list[dict[str, int]]:
    if not optimize:
        return [
            {
                "align_search": int(align_search),
                "corr_end_samples": int(corr_end_samples),
                "fit_end_samples": int(fit_end_samples),
            }
        ]

    align_vals = _parse_int_csv(align_search_grid)
    corr_vals = _parse_int_csv(corr_end_grid)
    fit_vals = _parse_int_csv(fit_end_grid)
    out: list[dict[str, int]] = []
    for a, c, f in itertools.product(align_vals, corr_vals, fit_vals):
        if c <= 1 or f <= 1:
            continue
        out.append(
            {
                "align_search": int(a),
                "corr_end_samples": int(c),
                "fit_end_samples": int(f),
            }
        )
    if not out:
        raise ValueError("No valid parameter combinations generated.")
    return out


def _config_id(cfg: dict[str, int]) -> str:
    return (
        f"as{cfg['align_search']}_"
        f"c{cfg['corr_end_samples']}_"
        f"f{cfg['fit_end_samples']}"
    )


def run(
    dataset_root: Path,
    output_csv: Path,
    error_csv: Path,
    config_summary_csv: Path,
    near_distance_m: float,
    blank_samples: int,
    ringdown_window_end_s: float,
    align_search: int,
    corr_end_samples: int,
    fit_end_samples: int,
    optimize: bool,
    align_search_grid: str,
    corr_end_grid: str,
    fit_end_grid: str,
    only_distance_cm_csv: str | None,
    max_files: int | None,
    progress_every_files: int,
) -> None:
    ref_pickle = dataset_root / "referenz" / "referenz.pickle"
    if not ref_pickle.exists():
        raise FileNotFoundError(f"Reference pickle not found: {ref_pickle}")

    templates = _build_ref_templates(ref_pickle)

    configs = _build_configs(
        optimize=optimize,
        align_search=align_search,
        corr_end_samples=corr_end_samples,
        fit_end_samples=fit_end_samples,
        align_search_grid=align_search_grid,
        corr_end_grid=corr_end_grid,
        fit_end_grid=fit_end_grid,
    )
    config_ids = [_config_id(cfg) for cfg in configs]
    distance_filter_cm = _parse_optional_distance_cm_filter(only_distance_cm_csv)

    print(f"Configs: {len(configs)}")
    print("Config IDs:", ", ".join(config_ids))
    if distance_filter_cm is not None:
        print(f"Distance filter (cm): {sorted(distance_filter_cm)}")

    rows: list[dict] = []
    errors: list[dict] = []
    obj_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir() and d.name.lower() != "referenz"])
    total_files = 0
    processed_files = 0

    for obj_dir in obj_dirs:
        for pkl in sorted(obj_dir.glob("*.pickle")):
            total_files += 1
            if max_files is not None and processed_files >= max_files:
                break

            distance_m = _parse_distance_m(pkl.stem)
            if distance_m is None:
                errors.append(
                    {
                        "file": str(pkl),
                        "mic": "",
                        "trial": -1,
                        "error": "Distance parse failed from filename",
                    }
                )
                continue

            distance_cm = int(round(distance_m * 100.0))
            if distance_filter_cm is not None and distance_cm not in distance_filter_cm:
                continue

            trials = load_trials(pkl)
            processed_files += 1
            if progress_every_files > 0 and (processed_files % progress_every_files == 0):
                print(f"Processed files: {processed_files} / {total_files} seen")

            for mic, signals in trials.items():
                if mic not in templates:
                    errors.append(
                        {
                            "file": str(pkl),
                            "mic": mic,
                            "trial": -1,
                            "error": "Missing reference template for mic",
                        }
                    )
                    continue

                ref_template = templates[mic]
                for i, signal in enumerate(signals):
                    x = np.asarray(signal, dtype=float)

                    try:
                        y_blank = ringdown_handle(
                            x=x,
                            distance_m=near_distance_m,
                            ref_template=None,
                            blank_samples=blank_samples,
                            near_distance_m=near_distance_m,
                            return_diagnostics=False,
                        )
                        red_blank_db = ringdown_energy_db_reduction(
                            before=x,
                            after=y_blank,
                            fs_hz=2_000_000.0,
                            window_s=(0.0, ringdown_window_end_s),
                        )
                    except Exception as exc:
                        errors.append(
                            {
                                "file": str(pkl),
                                "mic": mic,
                                "trial": i,
                                "error": f"blank_baseline_failed: {repr(exc)}",
                            }
                        )
                        continue

                    for cfg in configs:
                        corr_end = int(cfg["corr_end_samples"])
                        fit_end = int(cfg["fit_end_samples"])
                        try:
                            y, diag = ringdown_handle(
                                x=x,
                                distance_m=distance_m,
                                ref_template=ref_template,
                                blank_samples=blank_samples,
                                near_distance_m=near_distance_m,
                                align_search=int(cfg["align_search"]),
                                align_corr_window=(0, corr_end),
                                fit_window=(0, fit_end),
                                return_diagnostics=True,
                            )

                            red_used_db = ringdown_energy_db_reduction(
                                before=x,
                                after=y,
                                fs_hz=2_000_000.0,
                                window_s=(0.0, ringdown_window_end_s),
                            )

                            used_template = bool(diag["used_template_subtraction"])
                            rows.append(
                                {
                                    "config_id": _config_id(cfg),
                                    "align_search": int(cfg["align_search"]),
                                    "corr_end_samples": corr_end,
                                    "fit_end_samples": fit_end,
                                    "object": obj_dir.name,
                                    "file": pkl.name,
                                    "distance_m": distance_m,
                                    "distance_cm": distance_cm,
                                    "mic": mic,
                                    "trial": i,
                                    "used_template_subtraction": used_template,
                                    "lag_samples": int(diag["lag_samples"]),
                                    "alpha": float(diag["alpha"]),
                                    "reduction_used_db": float(red_used_db),
                                    "reduction_blank_db": float(red_blank_db),
                                    "delta_vs_blank_db": float(red_used_db - red_blank_db),
                                }
                            )
                        except Exception as exc:
                            errors.append(
                                {
                                    "file": str(pkl),
                                    "mic": mic,
                                    "trial": i,
                                    "config_id": _config_id(cfg),
                                    "error": repr(exc),
                                }
                            )

        if max_files is not None and processed_files >= max_files:
            break

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_rows = pd.DataFrame(rows)
    df_errs = pd.DataFrame(errors)
    df_rows.to_csv(output_csv, index=False)
    df_errs.to_csv(error_csv, index=False)

    print(f"Investigated files: {total_files}")
    print(f"Processed files after filters: {processed_files}")
    print(f"Output rows: {len(df_rows)}")
    print(f"Errors: {len(df_errs)}")
    print(f"Summary CSV: {output_csv}")
    print(f"Error CSV: {error_csv}")

    if not df_rows.empty:
        near_mask = df_rows["distance_m"] < near_distance_m
        n_near = int(near_mask.sum())
        n_far = int((~near_mask).sum())
        print(f"Near-field rows (< {near_distance_m} m): {n_near}")
        print(f"Far-field rows (>= {near_distance_m} m): {n_far}")

        if n_near > 0:
            near_df = df_rows[near_mask]
            print(
                "Near-field reduction_used_db min/mean/median/max:",
                float(near_df["reduction_used_db"].min()),
                float(near_df["reduction_used_db"].mean()),
                float(near_df["reduction_used_db"].median()),
                float(near_df["reduction_used_db"].max()),
            )
            print(
                "Near-field delta_vs_blank_db min/mean/median/max:",
                float(near_df["delta_vs_blank_db"].min()),
                float(near_df["delta_vs_blank_db"].mean()),
                float(near_df["delta_vs_blank_db"].median()),
                float(near_df["delta_vs_blank_db"].max()),
            )

        grp = (
            df_rows.groupby(["config_id", "align_search", "corr_end_samples", "fit_end_samples"], as_index=False)
            .agg(
                rows=("config_id", "size"),
                mean_reduction_db=("reduction_used_db", "mean"),
                median_reduction_db=("reduction_used_db", "median"),
                min_reduction_db=("reduction_used_db", "min"),
                mean_delta_vs_blank_db=("delta_vs_blank_db", "mean"),
                median_delta_vs_blank_db=("delta_vs_blank_db", "median"),
                min_delta_vs_blank_db=("delta_vs_blank_db", "min"),
                template_usage_rate=("used_template_subtraction", "mean"),
            )
            .sort_values(["mean_delta_vs_blank_db", "mean_reduction_db"], ascending=False)
        )
        config_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        grp.to_csv(config_summary_csv, index=False)
        print(f"Config summary CSV: {config_summary_csv}")
        print("Top configs:")
        print(grp.head(10).to_string(index=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Investigate reference-template ringdown handling across a dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="Multifrequenz Dataset/Multifrequenz",
        help="Dataset root containing object folders and referenz/referenz.pickle",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/ringdown_reference_investigation.csv",
        help="Path to write per-trial metrics CSV",
    )
    parser.add_argument(
        "--error-csv",
        type=str,
        default="outputs/ringdown_reference_investigation_errors.csv",
        help="Path to write errors CSV",
    )
    parser.add_argument(
        "--config-summary-csv",
        type=str,
        default="outputs/ringdown_reference_config_summary.csv",
        help="Path to write per-config aggregated metrics",
    )
    parser.add_argument(
        "--near-distance-m",
        type=float,
        default=0.50,
        help="Distance threshold for template subtraction branch",
    )
    parser.add_argument(
        "--blank-samples",
        type=int,
        default=2400,
        help="Blanking length used by far-field branch",
    )
    parser.add_argument(
        "--ringdown-window-end-s",
        type=float,
        default=1.2e-3,
        help="End time for early-window energy-reduction metric",
    )
    parser.add_argument(
        "--align-search",
        type=int,
        default=40,
        help="Default alignment lag search range used in non-optimization mode",
    )
    parser.add_argument(
        "--corr-end-samples",
        type=int,
        default=900,
        help="Correlation window end sample; actual window is [0, corr-end)",
    )
    parser.add_argument(
        "--fit-end-samples",
        type=int,
        default=2400,
        help="Least-squares fit window end sample; actual window is [0, fit-end)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable grid search over alignment and fit parameters",
    )
    parser.add_argument(
        "--align-search-grid",
        type=str,
        default="20,40,60",
        help="CSV integers for align_search when --optimize is enabled",
    )
    parser.add_argument(
        "--corr-end-grid",
        type=str,
        default="600,900,1200",
        help="CSV integers for corr-end sample when --optimize is enabled",
    )
    parser.add_argument(
        "--fit-end-grid",
        type=str,
        default="1800,2400,3000",
        help="CSV integers for fit-end sample when --optimize is enabled",
    )
    parser.add_argument(
        "--only-distance-cm",
        type=str,
        default=None,
        help="Optional CSV of distances in cm to include, e.g. 25,50",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on processed pickle files after filters",
    )
    parser.add_argument(
        "--progress-every-files",
        type=int,
        default=10,
        help="Print progress every N processed files",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run(
        dataset_root=Path(args.dataset_root),
        output_csv=Path(args.output_csv),
        error_csv=Path(args.error_csv),
        config_summary_csv=Path(args.config_summary_csv),
        near_distance_m=float(args.near_distance_m),
        blank_samples=int(args.blank_samples),
        ringdown_window_end_s=float(args.ringdown_window_end_s),
        align_search=int(args.align_search),
        corr_end_samples=int(args.corr_end_samples),
        fit_end_samples=int(args.fit_end_samples),
        optimize=bool(args.optimize),
        align_search_grid=str(args.align_search_grid),
        corr_end_grid=str(args.corr_end_grid),
        fit_end_grid=str(args.fit_end_grid),
        only_distance_cm_csv=args.only_distance_cm,
        max_files=args.max_files,
        progress_every_files=int(args.progress_every_files),
    )


if __name__ == "__main__":
    main()
