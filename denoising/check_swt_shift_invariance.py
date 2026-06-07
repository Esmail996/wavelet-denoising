import numpy as np
import matplotlib.pyplot as plt
from denoising.wavelet_choice import optimal_wavelets, wavespace

np.random.seed(0)
# Synthetic 40 kHz burst
fs = 2_000_000.0
t = np.arange(0, 0.005, 1/fs)  # 10,000 samples
burst_center = 3000
burst = np.zeros_like(t)
idx = np.arange(burst_center - 400, burst_center + 400)
window = np.hanning(len(idx))
burst[idx] = window * np.sin(2 * np.pi * 40_000 * t[idx])
burst += 0.1 * np.random.randn(len(t))

burst_shifted = np.roll(burst, 1)

# Plot the signal
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
axes[0].plot(t * 1000, burst, lw=0.7)
axes[0].set_title("Synthetic 40 kHz burst (with noise)")
axes[0].set_ylabel("Amplitude")
axes[1].plot(t * 1000, burst_shifted, lw=0.7, color="orange")
axes[1].set_title("1-sample shifted burst")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlabel("Time (ms)")
plt.tight_layout()
plt.show()

wave_family = wavespace()
r1, _ = optimal_wavelets(burst, wave_family, nw=5)
r2, _ = optimal_wavelets(burst_shifted, wave_family, nw=5)

print("Original top-5:")
for r in r1:
    print(f"  {r['wavelet']:12s}  μ_sc = {r['mu_sc']:.6f}")
print("Shifted  top-5:")
for r in r2:
    print(f"  {r['wavelet']:12s}  μ_sc = {r['mu_sc']:.6f}")
print("μ_sc diff (top):", abs(r1[0]["mu_sc"] - r2[0]["mu_sc"]))
