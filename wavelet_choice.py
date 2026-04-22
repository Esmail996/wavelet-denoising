import numpy as np
import pywt
import pandas as pd
from pathlib import Path
import argparse

from data_loader import load_data_folder
from parameters import BOX_DATA_DIR, OUT_DIR


# ============================================================
# Replace this later with your actual signal from a .pickle file
# ============================================================
# Example placeholder:
signal = np.asarray([0.0, 1.0, 0.5, -0.2, 0.1, 0.0], dtype=float)

# Keep the same variable name the way you asked:
x = signal


# ============================================================
# wavespace.m equivalent
# ============================================================
def wavespace():
    """
    Candidate wavelet set, following the MATLAB code / paper style.
    """
    wave_family = []

    # Biorthogonal
    wave_family += ["bior1.1", "bior1.3", "bior1.5", "bior2.2", "bior2.4", "bior2.6"]

    # Coiflet
    wave_family += ["coif1", "coif2", "coif3", "coif4", "coif5"]

    # Daubechies
    wave_family += ["db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10", "db11"]

    # Reverse biorthogonal
    wave_family += ["rbio1.3", "rbio1.5", "rbio2.2", "rbio2.4", "rbio2.6", "rbio2.8"]

    # Symlet
    wave_family += ["sym2", "sym3", "sym4", "sym5", "sym6", "sym7"]

    # MATLAB code also mentioned dmey
    wave_family += ["dmey"]

    return wave_family


# ============================================================
# wavecoef.m equivalent
# ============================================================
def wavecoef(X, wave_family):
    """
    Compute detail coefficients for each wavelet.

    Returns
    -------
    approx_coef : dict
        approx_coef[wname] = cA_N
    det_coef : dict
        det_coef[wname] = [D1, D2, ..., DN] in ascending level order
    Nmax : int
        floor(log2(len(X))) as in MATLAB code
    """
    X = np.asarray(X, dtype=float).ravel()
    if len(X) < 2:
        raise ValueError("Signal must have length >= 2.")

    Nmax = int(np.floor(np.log2(len(X))))
    approx_coef = {}
    det_coef = {}

    for wname in wave_family:
        try:
            wavelet = pywt.Wavelet(wname)
        except Exception:
            # Skip unsupported wavelets on current installation
            continue

        # Use the MATLAB-style maximum based on signal length,
        # but ensure pywt can actually support that level.
        max_allowed = pywt.dwt_max_level(data_len=len(X), filter_len=wavelet.dec_len)
        level = min(Nmax, max_allowed)

        if level < 1:
            continue

        coeffs = pywt.wavedec(X, wavelet=wname, level=level, mode="symmetric")
        # coeffs format in pywt: [cA_L, cD_L, cD_{L-1}, ..., cD_1]
        cA = coeffs[0]
        details_desc = coeffs[1:]
        details_asc = details_desc[::-1]  # convert to [D1, D2, ..., DL]

        approx_coef[wname] = cA
        det_coef[wname] = details_asc

    return approx_coef, det_coef, Nmax


# ============================================================
# Effective decomposition level support (same paper idea)
# ============================================================
def effective_decomposition_levels(det_coef):
    """
    Compute effective decomposition level using Rj = LDj / Lf > 1.5
    as described in the paper.

    Returns
    -------
    edl_info : dict
        edl_info[wname] = {
            "filter_length": Lf,
            "ratios": [R1, R2, ...],
            "effective_level": edl
        }
    """
    edl_info = {}

    for wname, details in det_coef.items():
        wavelet = pywt.Wavelet(wname)
        Lf = wavelet.dec_len
        ratios = []

        for Dj in details:
            LDj = len(Dj)
            Rj = LDj / Lf
            ratios.append(Rj)

        # Keep levels where Rj > 1.5
        valid_levels = [j + 1 for j, r in enumerate(ratios) if r > 1.5]
        edl = max(valid_levels) if valid_levels else 0

        edl_info[wname] = {
            "filter_length": Lf,
            "ratios": ratios,
            "effective_level": edl,
        }

    return edl_info


# ============================================================
# Sparsity.m equivalent
# ============================================================
def sparsity(det_coef):
    """
    Compute sparsity for each wavelet and level:
        S_j = max(abs(D_j)) / sum(abs(D_j))

    Returns
    -------
    s : dict
        s[wname] = [S1, S2, ..., SL]
    """
    s = {}

    for wname, details in det_coef.items():
        s_levels = []
        for Dj in details:
            absDj = np.abs(np.asarray(Dj, dtype=float))
            denom = np.sum(absDj)

            if denom == 0:
                Sj = 0.0
            else:
                Sj = float(np.max(absDj) / denom)

            s_levels.append(Sj)

        s[wname] = s_levels

    return s


# ============================================================
# SparsityChange.m equivalent
# ============================================================
def sparsity_change(s):
    """
    Compute sparsity change:
        ΔS1 = 0
        ΔSj = Sj - S(j-1)

    Returns
    -------
    sc : dict
        sc[wname] = [ΔS1, ΔS2, ..., ΔSL]
    """
    sc = {}

    for wname, s_levels in s.items():
        delta = []
        for j in range(len(s_levels)):
            if j == 0:
                delta.append(0.0)
            else:
                delta.append(float(s_levels[j] - s_levels[j - 1]))
        sc[wname] = delta

    return sc


# ============================================================
# Decomlevel.m equivalent
# ============================================================
def decomlevel(sc, threshold=0.05):
    """
    Find κ using the MATLAB/paper rule:
    first level j where ΔSj > 0.05, then κ = j - 1

    Notes
    -----
    Levels in theory are 1-based.
    In Python list indices are 0-based, so careful conversion is used.

    Returns
    -------
    dl : dict
        dl[wname] = kappa
    """
    dl = {}

    for wname, delta in sc.items():
        kappa = 1  # fallback similar spirit to MATLAB handling
        found = False

        # j_theory runs from 2..L
        for idx in range(1, len(delta)):
            if delta[idx] > threshold:
                j_theory = idx + 1      # convert Python index to 1-based level
                kappa = j_theory - 1
                found = True
                break

        if not found:
            # No abrupt jump found
            # keep conservative fallback
            kappa = 1

        dl[wname] = kappa

    return dl


# ============================================================
# Meansc.m equivalent
# ============================================================
def meansc(s, dl):
    """
    Compute mean of sparsity change:
        μsc = (S_(κ+1) - S_1) / (κ - 1)

    MATLAB code had a special handling when κ == 1.

    Returns
    -------
    msc : dict
        msc[wname] = μsc
    """
    msc = {}

    for wname, s_levels in s.items():
        kappa = dl[wname]

        # Need S1 and S_(kappa+1)
        # Python indices: S1 -> s_levels[0], S_(k+1) -> s_levels[kappa]
        if len(s_levels) == 0:
            msc[wname] = np.nan
            continue

        if kappa == 1:
            # Same spirit as MATLAB special-case handling
            msc[wname] = float(s_levels[0])
            continue

        if kappa >= len(s_levels):
            # If κ+1 would exceed available levels, clamp safely
            # This keeps the code robust without changing the method's intention.
            kappa = len(s_levels) - 1

        numerator = s_levels[kappa] - s_levels[0]
        denominator = kappa - 1

        if denominator == 0:
            msc[wname] = float(s_levels[0])
        else:
            msc[wname] = float(numerator / denominator)

    return msc


# ============================================================
# optimalwavelets.m equivalent
# ============================================================
def optimal_wavelets(X, wave_family, nw=5):
    """
    MATLAB-style pipeline:
      1) wavecoef
      2) effective decomposition level (computed as in paper)
      3) sparsity
      4) sparsity_change
      5) decomlevel
      6) meansc
      7) sort descending and return top nw wavelets

    Returns
    -------
    results : list of dict
        Each dict contains:
          - wavelet
          - kappa
          - mu_sc
          - effective_level
          - filter_length
          - sparsity
          - sparsity_change
          - ratios
    extras : dict
        Contains intermediate objects for inspection.
    """
    _, det_coef, Nmax = wavecoef(X, wave_family)
    edl_info = effective_decomposition_levels(det_coef)
    s = sparsity(det_coef)
    sc = sparsity_change(s)
    dl = decomlevel(sc, threshold=0.05)
    msc = meansc(s, dl)

    rows = []
    for wname in msc.keys():
        rows.append({
            "wavelet": wname,
            "kappa": dl[wname],
            "mu_sc": msc[wname],
            "effective_level": edl_info[wname]["effective_level"],
            "filter_length": edl_info[wname]["filter_length"],
            "sparsity": s[wname],
            "sparsity_change": sc[wname],
            "ratios": edl_info[wname]["ratios"],
        })

    rows = sorted(
        rows,
        key=lambda r: (-np.inf if np.isnan(r["mu_sc"]) else r["mu_sc"]),
        reverse=True
    )

    results = rows[:nw]

    extras = {
        "Nmax_floor_log2_lenX": Nmax,
        "det_coef": det_coef,
        "edl_info": edl_info,
        "sparsity": s,
        "sparsity_change": sc,
        "decomlevel": dl,
        "meansc": msc,
    }

    return results, extras


# ============================================================
# Optional helper: SNR in the paper's style
# ============================================================
def compute_snr_paper_style(x, signal_peak_index=None, noise_region=None):
    """
    Paper-style SNR:
        SNR = Signal_peak / Noise_rms

    Parameters
    ----------
    x : array-like
        Input signal
    signal_peak_index : int or None
        If given, use abs(x[signal_peak_index]) as Signal_peak.
        If None, use max(abs(x)).
    noise_region : tuple(start, end) or None
        Region assumed to contain only noise.
        If None, use first 200 samples (or shorter if signal is shorter),
        matching the paper's idea for simulated data.

    Returns
    -------
    snr_linear : float
        Signal_peak / Noise_rms
    snr_db : float
        20*log10(snr_linear)
    signal_peak : float
    noise_rms : float
    """
    x = np.asarray(x, dtype=float).ravel()
    if len(x) == 0:
        raise ValueError("Signal is empty.")

    if signal_peak_index is None:
        signal_peak = float(np.max(np.abs(x)))
    else:
        signal_peak = float(np.abs(x[signal_peak_index]))

    if noise_region is None:
        end = min(200, len(x))
        noise_seg = x[:end]
    else:
        start, end = noise_region
        noise_seg = x[start:end]

    if len(noise_seg) == 0:
        raise ValueError("Noise region is empty.")

    noise_rms = float(np.sqrt(np.mean(noise_seg ** 2)))

    if noise_rms == 0:
        snr_linear = np.inf
        snr_db = np.inf
    else:
        snr_linear = signal_peak / noise_rms
        snr_db = 20.0 * np.log10(snr_linear)

    return snr_linear, snr_db, signal_peak, noise_rms


def run_method_on_dataset(
    box_dir,
    mic="mic2",
    nw=5,
    max_rows=None,
    output_dir=None,
):
    """
    Apply the optimal wavelet method to all rows in the dataset folder.

    Parameters
    ----------
    box_dir : str or Path
        Folder containing measurement .pickle files.
    mic : str
        One of: mic1, mic2, mic3.
    nw : int
        Number of top candidates to keep per row.
    max_rows : int or None
        If provided, process only first max_rows rows (for quick runs).
    output_dir : str or Path or None
        Destination folder for CSV outputs.

    Returns
    -------
    df_topn : pd.DataFrame
        One row per (signal, rank).
    df_best : pd.DataFrame
        One row per signal using best-ranked wavelet.
    df_summary : pd.DataFrame
        Summary grouped by selected best wavelet.
    """
    mic = mic.lower()
    if mic not in {"mic1", "mic2", "mic3"}:
        raise ValueError("mic must be one of: mic1, mic2, mic3")

    box_dir = Path(box_dir)
    if output_dir is None:
        output_dir = Path(OUT_DIR) / "wavelet_choice"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data_folder(box_dir)
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows > 0:
            df = df.iloc[:max_rows].copy()

    wave_family = wavespace()
    topn_rows = []
    best_rows = []

    for i, row in df.iterrows():
        sig = np.asarray(row[mic], dtype=float)

        # Light preprocessing to stabilize coefficient statistics.
        sig = sig - np.mean(sig)

        results, _ = optimal_wavelets(sig, wave_family, nw=nw)
        if not results:
            continue

        snr_linear, snr_db, signal_peak, noise_rms = compute_snr_paper_style(sig)

        for rank, r in enumerate(results, start=1):
            topn_rows.append({
                "row_index": int(i),
                "file": row["file"],
                "distance_cm": int(row["distance_cm"]),
                "angle_deg": int(row["angle_deg"]),
                "meas_idx": int(row["meas_idx"]),
                "mic": mic,
                "rank": rank,
                "wavelet": r["wavelet"],
                "kappa": int(r["kappa"]),
                "mu_sc": float(r["mu_sc"]),
                "effective_level": int(r["effective_level"]),
                "filter_length": int(r["filter_length"]),
                "snr_linear": float(snr_linear),
                "snr_db": float(snr_db),
                "signal_peak": float(signal_peak),
                "noise_rms": float(noise_rms),
            })

        best = results[0]
        best_rows.append({
            "row_index": int(i),
            "file": row["file"],
            "distance_cm": int(row["distance_cm"]),
            "angle_deg": int(row["angle_deg"]),
            "meas_idx": int(row["meas_idx"]),
            "mic": mic,
            "best_wavelet": best["wavelet"],
            "kappa": int(best["kappa"]),
            "mu_sc": float(best["mu_sc"]),
            "effective_level": int(best["effective_level"]),
            "snr_db": float(snr_db),
        })

    df_topn = pd.DataFrame(topn_rows)
    df_best = pd.DataFrame(best_rows)

    if df_best.empty:
        df_summary = pd.DataFrame(
            columns=["best_wavelet", "count", "mean_mu_sc", "mean_kappa", "mean_effective_level", "mean_snr_db"]
        )
    else:
        df_summary = (
            df_best.groupby("best_wavelet", as_index=False)
            .agg(
                count=("best_wavelet", "size"),
                mean_mu_sc=("mu_sc", "mean"),
                mean_kappa=("kappa", "mean"),
                mean_effective_level=("effective_level", "mean"),
                mean_snr_db=("snr_db", "mean"),
            )
            .sort_values(["count", "mean_mu_sc"], ascending=[False, False])
            .reset_index(drop=True)
        )

    topn_path = output_dir / "per_signal_topn.csv"
    best_path = output_dir / "per_signal_best.csv"
    summary_path = output_dir / "summary_best_wavelet.csv"

    df_topn.to_csv(topn_path, index=False)
    df_best.to_csv(best_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    print(f"Processed rows: {len(df_best)}")
    print(f"Top-N output : {topn_path}")
    print(f"Best output  : {best_path}")
    print(f"Summary      : {summary_path}")

    return df_topn, df_best, df_summary


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Wavelet choice on full dataset")
    parser.add_argument("--box-dir", type=str, default=str(BOX_DATA_DIR), help="Path to dataset folder with .pickle files")
    parser.add_argument("--mic", type=str, default="mic2", choices=["mic1", "mic2", "mic3"], help="Microphone column")
    parser.add_argument("--top-n", type=int, default=5, help="Top-N wavelets per signal")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory")
    parser.add_argument(
        "--single-demo",
        action="store_true",
        help="Run the original single-signal demo instead of dataset mode",
    )
    return parser


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    if args.single_demo:
        wave_family = wavespace()
        results, _ = optimal_wavelets(x, wave_family, nw=args.top_n)

        print("\nTop optimal wavelets:")
        for r in results:
            print(
                f"wavelet={r['wavelet']:8s}  "
                f"kappa={r['kappa']:2d}  "
                f"mu_sc={r['mu_sc']:.6f}  "
                f"effective_level={r['effective_level']:2d}"
            )

        snr_linear, snr_db, signal_peak, noise_rms = compute_snr_paper_style(x)
        print("\nPaper-style SNR:")
        print(f"Signal_peak = {signal_peak:.6f}")
        print(f"Noise_rms   = {noise_rms:.6f}")
        print(f"SNR         = {snr_linear:.6f}")
        print(f"SNR_dB      = {snr_db:.6f}")
    else:
        run_method_on_dataset(
            box_dir=args.box_dir,
            mic=args.mic,
            nw=args.top_n,
            max_rows=args.max_rows,
            output_dir=args.output_dir,
        )