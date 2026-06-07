#!/usr/bin/env python3
"""
make_roi_figures.py - figures for the ROI Extraction chapter.

  A. fig_roi_window_design   - STFT windows (variable, peak at the back) vs the
                               proposed fixed peak-centred window
  B. fig_roi_energy_extent   - echo energy captured vs samples before/after the
                               peak, justifying the [-256, +768] window
  C. fig_echo_window         - aligned-echo envelope profile with the window and
                               the secondary-echo region marked

Requires: numpy pandas pywavelets scipy matplotlib, the dataset (DATA_DIR), and
the TOF peaks CSV (PEAKS_CSV with column peak_idx_path_b).
"""
import sys, numpy as np, pandas as pd, pywt, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATA_DIR  = "data_full/Multifrequenz"
PEAKS_CSV = "/mnt/user-data/uploads/multifrequenz_denoised_branchA_bayes_garrote_peaks.csv"
FS = 2_000_000.0
WIN = {25:(2610,3498), 50:(5461,6184), 75:(8437,9120), 100:(11298,12123), 125:(14063,15135)}
NULL = (600, 1488); ANGLES = [-10,-5,0,5,10]; CARRIERS = [40e3,50e3,60e3]
BEFORE, AFTER = 256, 768                      # the chosen fixed window
COL = {"Box":"#c1432e","Dose":"#2f6fb0","Glas":"#3f9153"}
LAGL, LAGR = 900, 1500


def cwt_env(x, fc):
    c,_ = pywt.cwt(x, [FS/fc], "cmor1.5-1.0", sampling_period=1.0/FS); return np.abs(c[0])

def save(fig, name):
    fig.savefig(f"{name}.pdf"); fig.savefig(f"{name}.png", dpi=140); plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# --- A: window design (STFT vs fixed peak-centred) ------------------------
def fig_window_design():
    print("Figure A: window design")
    pk = pd.read_csv(PEAKS_CSV)
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
    dists = sorted(WIN)
    for i, d in enumerate(dists):
        lo, hi = WIN[d]
        s = pk[pk.distance_cm == d]["peak_idx_path_b"]
        p1, p50, p99 = np.percentile(s, [1, 50, 99])
        # (a) STFT window bar + peak band
        ax[0].barh(i, hi-lo, left=lo, height=0.55, color="#ccd5e0", edgecolor="#7a8aa0")
        ax[0].barh(i, p99-p1, left=p1, height=0.22, color="#2f6fb0", alpha=0.7)
        ax[0].plot(p50, i, "|", color="k", ms=12, mew=2)
        ax[0].text(hi+150, i, f"w={hi-lo}", va="center", fontsize=8)
        # (b) fixed peak-centred window (same width, peak at fixed offset)
        c0 = p50
        ax[1].barh(i, BEFORE+AFTER, left=c0-BEFORE, height=0.55, color="#cfe3d4", edgecolor="#3f9153")
        ax[1].plot(c0, i, "|", color="k", ms=12, mew=2)
    for a, ttl in zip(ax, ["(a) STFT search windows: variable size, peak at the back",
                           f"(b) Fixed peak-centred window [-{BEFORE}, +{AFTER}] (1024 samples)"]):
        a.set_yticks(range(len(dists))); a.set_yticklabels([f"{d} cm" for d in dists])
        a.set_xlabel("sample index (fs = 2 MHz)"); a.set_title(ttl, fontsize=10)
        a.set_xlim(0, 16500); a.grid(axis="x", alpha=0.25)
    ax[0].plot([], [], color="#2f6fb0", lw=6, alpha=0.7, label="peak p1\u2013p99")
    ax[0].plot([], [], "|", color="k", ms=10, mew=2, label="peak median")
    ax[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); save(fig, "fig_roi_window_design")


# --- collect aligned, peak-normalised envelopes (for B and C) -------------
def collect(fc, mic="Mic2", step=10):
    stk = []
    for o in ["Box","Dose","Glas"]:
        for d in WIN:
            lo, hi = WIN[d]
            for a in ANGLES:
                df = pd.read_pickle(f"{DATA_DIR}/{o}/{d}cm_{a}Grad.pickle")
                for tr in range(0, 100, step):
                    x = np.asarray(df[mic].iloc[tr], float); e = cwt_env(x, fc)
                    k = lo + int(np.argmax(e[lo:hi]))
                    if k-LAGL < 0 or k+LAGR >= len(e): continue
                    stk.append(e[k-LAGL:k+LAGR] / (e[k] + 1e-12))
    return np.array(stk)


# --- B: echo-energy extent before/after the peak --------------------------
def fig_energy_extent(stack=None):
    print("Figure B: echo-energy extent")
    S = collect(50e3) if stack is None else stack
    c = LAGL
    base = np.median(S[:, :200], axis=1, keepdims=True)
    P = np.clip(S - base, 0, None) ** 2                       # echo energy above noise
    before = np.cumsum(P[:, :c+1][:, ::-1], axis=1) / (P[:, :c+1].sum(1, keepdims=True) + 1e-12)
    after  = np.cumsum(P[:, c:], axis=1)         / (P[:, c:].sum(1, keepdims=True) + 1e-12)
    bmed, amed = np.median(before, 0) * 100, np.median(after, 0) * 100
    lag_b = np.arange(before.shape[1]); lag_a = np.arange(after.shape[1])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(-lag_b, bmed, color="#2f6fb0", lw=1.6, label="before the peak")
    ax.plot(lag_a, amed, color="#c1432e", lw=1.6, label="after the peak")
    for x0, c0, lab in [(-BEFORE, "#2f6fb0", f"-{BEFORE}"), (AFTER, "#c1432e", f"+{AFTER}")]:
        ax.axvline(x0, color=c0, ls="--", lw=1)
        y = np.interp(abs(x0), lag_b if x0 < 0 else lag_a, bmed if x0 < 0 else amed)
        ax.plot(x0, y, "o", color=c0); ax.annotate(f"{lab} samples\n{y:.0f}% captured",
                 (x0, y), textcoords="offset points", xytext=(8 if x0>0 else -8, -18),
                 ha="left" if x0>0 else "right", fontsize=8, color=c0)
    ax.axhline(95, color="#888", ls=":", lw=0.8); ax.text(-880, 96, "95%", fontsize=8, color="#666")
    ax.set_xlabel("samples from the peak"); ax.set_ylabel("echo energy captured (%)")
    ax.set_title("Echo energy captured vs window extent (50 kHz, Mic2, all 75 cells)", fontsize=10)
    ax.set_xlim(-LAGL, LAGR); ax.set_ylim(40, 101); ax.legend(loc="lower right"); ax.grid(alpha=0.25)
    fig.tight_layout(); save(fig, "fig_roi_energy_extent")


# --- C: aligned-echo profile with the window ------------------------------
def fig_echo_window():
    print("Figure C: echo window profile")
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for j, (fc, col) in enumerate([(40e3, "#c1432e"), (50e3, "#2f6fb0")]):
        S = collect(fc)
        lag = np.arange(-LAGL, LAGR); us = lag * 0.5
        med = np.median(S, 0); p90 = np.percentile(S, 90, 0)
        ax[j].fill_between(us, 0, p90, color=col, alpha=0.12, label="90th-percentile envelope")
        ax[j].plot(us, med, color=col, lw=1.4, label="median envelope")
        ax[j].axvline(0, color="k", lw=0.8, ls=":")
        ax[j].axvspan(-BEFORE*0.5, AFTER*0.5, color="#3f9153", alpha=0.10)
        ax[j].axvline(-BEFORE*0.5, color="#2c6e49", lw=1.2); ax[j].axvline(AFTER*0.5, color="#2c6e49", lw=1.2)
        ax[j].annotate(f"window [-{BEFORE}, +{AFTER}]\n(-{BEFORE*0.5:.0f} to +{AFTER*0.5:.0f} us)",
                       (AFTER*0.5-8, 0.92), fontsize=8, ha="right", color="#2c6e49")
        ax[j].axvspan(500*0.5, 1150*0.5, color="#d6a32f", alpha=0.13)
        ax[j].annotate("far secondary echo\n(Dose ~ +505 us, excluded)", (980*0.5, 0.30),
                       fontsize=7.5, ha="center", color="#8a6000")
        ax[j].set_title(f"{int(fc/1e3)} kHz CWT envelope, aligned on the peak", fontsize=10)
        ax[j].set_xlabel("lag from peak (us)"); ax[j].set_ylabel("envelope (norm. to peak)")
        ax[j].set_xlim(-250, 800); ax[j].set_ylim(0, 1.05); ax[j].legend(fontsize=8, loc="upper right")
    fig.tight_layout(); save(fig, "fig_echo_window")


if __name__ == "__main__":
    funcs = {"A": fig_window_design, "B": fig_energy_extent, "C": fig_echo_window}
    which = sys.argv[1:] or ["A", "B", "C"]
    for k in which:
        funcs[k]()
    print("done.")
