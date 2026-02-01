"""
Wavelet Denoising Application
============================

This application demonstrates wavelet-based signal denoising on real sensor data.

Author: Your Name
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Plot the raw signal
plt.figure(figsize=(12, 6))
plt.plot(signal, 'b-', alpha=0.7)
plt.title("Raw Mic2 Signal (Reference Measurement)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

# Save plot instead of showing (since we're in terminal)
plt.savefig('raw_signal.png', dpi=150, bbox_inches='tight')
print("✅ Plot saved as 'raw_signal.png'")

# Now let's apply denoising
print("\n🔬 Applying wavelet denoising...")
from denoising import denoise

# Test different methods
methods = ['universal', 'stein', 'energy']
denoised_signals = {}

for method in methods:
    print(f"   Denoising with {method} method...")
    try:
        denoised = denoise(signal,
                          wavelet='db3',
                          level=4,
                          method=method,
                          thr_mode='soft')
        denoised_signals[method] = denoised

        # Calculate noise reduction
        original_std = np.std(signal)
        denoised_std = np.std(denoised)
        noise_reduction = ((original_std - denoised_std) / original_std) * 100

        print(".4f"
              ".1f")

    except Exception as e:
        print(f"   ❌ Error with {method}: {e}")

# Create comparison plot
plt.figure(figsize=(15, 10))

# Original signal
plt.subplot(2, 2, 1)
plt.plot(signal[:2000], 'b-', alpha=0.7)  # Show first 2000 samples
plt.title("Original Signal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

# Denoised signals
for i, (method, denoised) in enumerate(denoised_signals.items()):
    plt.subplot(2, 2, i+2)
    plt.plot(denoised[:2000], 'r-', alpha=0.7)
    plt.title(f"Denoised - {method.capitalize()} Method")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('denoising_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Comparison plot saved as 'denoising_comparison.png'")

print("\n✅ Analysis complete!")
print("📊 Generated files:")
print("   - raw_signal.png: Original signal plot")
print("   - denoising_comparison.png: Before/after denoising comparison")