"""
Wavelet Denoising Application
============================

This application demonstrates wavelet-based signal denoising on real sensor data.

Author: Esmail Wahoud
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def bandpass(x, fs, f0, bw=3000, order=4):
    low = (f0 - bw) / (fs / 2)
    high = (f0 + bw) / (fs / 2)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)

# =====================================================================
# STEP 1: READ RAW DATA
# =====================================================================
print("=" * 60)
print("STEP 1: READING RAW DATA")
print("=" * 60)

# Load reference data
print("Loading reference data...")
df = pd.read_pickle("./Multifrequenz Dataset/Multifrequenz/referenz/referenz.pickle")
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Take one measurement
row = df.iloc[0]
print(f"First row keys: {list(row.keys()) if hasattr(row, 'keys') else 'N/A'}")

# Extract Mic2 signal
signal = np.array(row["Mic2"])
print(f"Signal shape: {signal.shape}")
print(".4f")
print(".4f")

# =====================================================================
# STEP 2: SAVE AND SHOW RAW SIGNAL
# =====================================================================
print("\n" + "=" * 60)
print("STEP 2: SAVING AND SHOWING RAW SIGNAL")
print("=" * 60)

# Plot the raw signal
plt.figure(figsize=(12, 6))
plt.plot(signal, 'b-', alpha=0.7)
plt.title("Raw Mic2 Signal (Reference Measurement)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

# Save plot
plt.savefig('raw_signal.png', dpi=150, bbox_inches='tight')
print("✅ Raw signal plot saved as 'raw_signal.png'")

# Save raw signal data
np.save('raw_signal.npy', signal)
print("✅ Raw signal data saved as 'raw_signal.npy'")

# =====================================================================
# STEP 3: APPLY BANDPASS FILTERS
# =====================================================================
print("\n" + "=" * 60)
print("STEP 3: APPLYING BANDPASS FILTERS")
print("=" * 60)

fs = 200_000  # sampling frequency

# Apply bandpass filters at different frequencies
print("Applying bandpass filters...")
sig_40k = bandpass(signal, fs, 40_000)
sig_50k = bandpass(signal, fs, 50_000)
sig_60k = bandpass(signal, fs, 60_000)

print("Applied bandpass filters:")
print(".4f")
print(".4f")
print(".4f")

# =====================================================================
# STEP 4: SAVE AND SHOW FILTERED SIGNALS
# =====================================================================
print("\n" + "=" * 60)
print("STEP 4: SAVING AND SHOWING FILTERED SIGNALS")
print("=" * 60)

# Create comparison plot of filtered signals
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(signal, 'b-', alpha=0.7)
plt.title("Original Signal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(sig_40k, 'g-', alpha=0.7)
plt.title("40kHz Bandpass Filtered")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(sig_50k, 'r-', alpha=0.7)
plt.title("50kHz Bandpass Filtered")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(sig_60k, 'm-', alpha=0.7)
plt.title("60kHz Bandpass Filtered")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('filtered_signals.png', dpi=150, bbox_inches='tight')
print("✅ Filtered signals plot saved as 'filtered_signals.png'")

# Save filtered signals
np.save('filtered_signals.npy', np.array([sig_40k, sig_50k, sig_60k]))
print("✅ Filtered signals data saved as 'filtered_signals.npy'")

# =====================================================================
# STEP 5: APPLY DENOISING
# =====================================================================
print("\n" + "=" * 60)
print("STEP 5: APPLYING WAVELET DENOISING")
print("=" * 60)

from denoising import denoise

# Test different methods on all signals
methods = ['universal', 'stein', 'energy']
signals_to_test = {
    'original': signal,
    '40k_filtered': sig_40k,
    '50k_filtered': sig_50k,
    '60k_filtered': sig_60k
}

all_results = {}

print("Testing denoising methods...")
for sig_name, sig_data in signals_to_test.items():
    print(f"\n{sig_name.upper()} SIGNAL:")
    all_results[sig_name] = {}

    for method in methods:
        print(f"   Denoising with {method} method...")
        try:
            denoised = denoise(sig_data,
                              wavelet='db3',
                              level=4,
                              method=method,
                              thr_mode='soft')
            all_results[sig_name][method] = denoised

            # Calculate noise reduction
            original_std = np.std(sig_data)
            denoised_std = np.std(denoised)
            noise_reduction = ((original_std - denoised_std) / original_std) * 100

            print(".4f"
                  ".1f")

        except Exception as e:
            print(f"   ❌ Error with {method}: {e}")

# =====================================================================
# STEP 6: SAVE DENOISING RESULTS
# =====================================================================
print("\n" + "=" * 60)
print("STEP 6: SAVING DENOISING RESULTS")
print("=" * 60)

# Create comprehensive comparison plot
plt.figure(figsize=(20, 15))

# Plot original and filtered signals (first row)
plt.subplot(5, 4, 1)
plt.plot(signal, 'b-', alpha=0.7)
plt.title("Original Signal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

plt.subplot(5, 4, 2)
plt.plot(sig_40k, 'g-', alpha=0.7)
plt.title("40kHz Filtered")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

plt.subplot(5, 4, 3)
plt.plot(sig_50k, 'r-', alpha=0.7)
plt.title("50kHz Filtered")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

plt.subplot(5, 4, 4)
plt.plot(sig_60k, 'm-', alpha=0.7)
plt.title("60kHz Filtered")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

# Plot denoised signals for each method
method_colors = {'universal': 'blue', 'stein': 'red', 'energy': 'green'}
row = 1
for method in methods:
    col = 0
    for sig_name, sig_data in signals_to_test.items():
        plt.subplot(5, 4, (row+1)*4 + col + 1)
        plt.plot(all_results[sig_name][method], color=method_colors[method], alpha=0.7)
        plt.title(f"{sig_name}\n{method.capitalize()}")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        col += 1
    row += 1

plt.tight_layout()
plt.savefig('denoising_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Denoising comparison plot saved as 'denoising_comparison.png'")

# Save denoising results
np.save('denoising_results.npy', all_results)
print("✅ Denoising results saved as 'denoising_results.npy'")

print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE!")
print("=" * 60)
print("📊 Generated files:")
print("   - raw_signal.png: Original signal plot")
print("   - raw_signal.npy: Original signal data")
print("   - filtered_signals.png: Bandpass filtered signals plot")
print("   - filtered_signals.npy: Filtered signals data")
print("   - denoising_comparison.png: Complete denoising comparison")
print("   - denoising_results.npy: All denoising results data")
print("\n🔍 Analysis performed on:")
print("   - Original signal")
print("   - 40kHz, 50kHz, 60kHz bandpass filtered signals")
print("   - Each tested with universal, stein, and energy denoising methods")