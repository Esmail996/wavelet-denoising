"""Synthetic-echo verification of the PRODUCTION estimator (pipeline2.estimate_max:
STFT-measured window + plain-maximum picker + parabolic sub-sample)."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from TOF_estimation import FS_HZ, SOUND_SPEED_M_S, CARRIERS_HZ
from pipeline2 import estimate_max

# production STFT-measured per-distance windows
PROD_WINDOWS = {25:(2610,3498), 50:(5461,6184), 75:(8437,9120),
                100:(11298,12123), 125:(14063,15135)}
N_SAMPLES=18000; SIGMA_CYCLES=4.0; MIC="Mic2"
MM_PER_SAMPLE=SOUND_SPEED_M_S/FS_HZ*1000.0/2.0; NS_PER_SAMPLE=1e9/FS_HZ
TX_OF_CARRIER={40_000.0:"Tx1",50_000.0:"Tx5",60_000.0:"Tx8"}
RNG=np.random.default_rng(20240601)

def synth_echo(n0, fc, amp=1.0, n_samples=N_SAMPLES):
    n=np.arange(n_samples,dtype=np.float64); sigma=SIGMA_CYCLES*FS_HZ/fc
    return amp*np.exp(-0.5*((n-n0)/sigma)**2)*np.sin(2*np.pi*fc*(n-n0)/FS_HZ)

def recovered(path, buf, dist_cm, tx):
    ta, tb, _ = estimate_max(buf, MIC, tx, PROD_WINDOWS[dist_cm])
    return (ta if path=="A" else tb)*FS_HZ

print("="*78); print("TEST 1  Noise-free recovery of known arrival (production estimator)"); print("="*78)
print(f"{'path':>4} {'carrier':>8} {'bias_ns':>9} {'bias_mm':>9} {'maxerr_ns':>10} {'maxerr_mm':>10}")
fracs=np.linspace(0.0,0.9,10)
for path in ("A","B"):
    for fc,tx in sorted(TX_OF_CARRIER.items()):
        errs=[]
        for d,(lo,hi) in PROD_WINDOWS.items():
            base=(lo+hi)//2
            for f in fracs:
                n0=base+f; errs.append(recovered(path,synth_echo(n0,fc),d,tx)-n0)
        e=np.asarray(errs)
        print(f"{path:>4} {fc/1000:>6.0f}k {e.mean()*NS_PER_SAMPLE:>9.2f} {e.mean()*MM_PER_SAMPLE:>9.4f} "
              f"{np.abs(e).max()*NS_PER_SAMPLE:>10.2f} {np.abs(e).max()*MM_PER_SAMPLE:>10.4f}")

print("\n"+"="*78); print("TEST 2  Timing bias and jitter vs SNR (150 realisations/cell)"); print("="*78)
SNRS_DB=[np.inf,30.,20.,12.,6.,0.]; N_REAL=150; DISTS=[25,75,125]; AMP=1.0
print(f"{'path':>4} {'SNR_dB':>7} {'bias_mm':>9} {'jitter_mm':>10} {'RMS_mm':>9}")
rms={"A":[],"B":[]}
for path in ("A","B"):
    for snr in SNRS_DB:
        errs=[]
        for d in DISTS:
            lo,hi=PROD_WINDOWS[d]; base=(lo+hi)//2
            for fc,tx in sorted(TX_OF_CARRIER.items()):
                clean=synth_echo(base+0.37,fc,amp=AMP)
                sig=0.0 if np.isinf(snr) else AMP/(10**(snr/20))
                for _ in range(N_REAL if not np.isinf(snr) else 1):
                    buf=clean+(sig*RNG.standard_normal(N_SAMPLES) if sig else 0.0)
                    errs.append(recovered(path,buf,d,tx)-(base+0.37))
        e=np.asarray(errs)
        rms[path].append(np.sqrt(np.mean(e**2))*MM_PER_SAMPLE)
        lab="inf" if np.isinf(snr) else f"{snr:.0f}"
        print(f"{path:>4} {lab:>7} {e.mean()*MM_PER_SAMPLE:>9.4f} {e.std()*MM_PER_SAMPLE:>10.4f} {np.sqrt(np.mean(e**2))*MM_PER_SAMPLE:>9.4f}")

xs=[40 if np.isinf(s) else s for s in SNRS_DB]
fig,ax=plt.subplots(figsize=(6.2,4.0))
ax.plot(xs,rms["A"],"o-",label="Path A (matched filter)")
ax.plot(xs,rms["B"],"s-",label="Path B (Morlet CWT)")
ax.set_xticks(xs); ax.set_xticklabels(["inf","30","20","12","6","0"]); ax.invert_xaxis()
ax.set_xlabel("peak-to-noise SNR (dB)"); ax.set_ylabel("RMS timing error (mm of range)")
ax.set_title("Synthetic-echo timing error vs SNR (production estimator)")
ax.grid(True,alpha=0.3); ax.legend(); fig.tight_layout()
fig.savefig("fig_synthetic_tof_snr.pdf",bbox_inches="tight"); fig.savefig("fig_synthetic_tof_snr.png",dpi=150,bbox_inches="tight")
print("\nwrote fig_synthetic_tof_snr")
