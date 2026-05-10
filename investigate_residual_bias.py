from pathlib import Path
import numpy as np
import pandas as pd
from TOF_estimation import SOUND_SPEED_M_S, FS_HZ, CARRIERS_HZ, tof_to_distance_calibrated

csv_path = Path('outputs/tof_matched/multifrequenz_denoised_branchA_sym6.csv')
df = pd.read_csv(csv_path)

df['err_b_cal'] = df.apply(
    lambda r: 100.0 * tof_to_distance_calibrated(r['tof_path_b_s'], r['mic'], r['tx'], 'path_b') - r['distance_cm'],
    axis=1,
)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Cycle-skip outliers on Mic1 at short range
# ─────────────────────────────────────────────────────────────────────────────
print('=' * 90)
print('PART 1 — Cycle-skip outliers on Mic1 (Path B, calibrated)')
print('=' * 90)

print('\nCycle-skip distance equivalent per Tx (1 period at carrier):')
for tx, fc in sorted(CARRIERS_HZ.items()):
    period_s = 1.0 / fc
    delta_d_cm = SOUND_SPEED_M_S * period_s / 2 * 100
    print(f'  {tx} ({fc/1000:.0f} kHz): 1-cycle skip ≈ {delta_d_cm:.2f} cm  ({period_s*1e6:.1f} µs)')

SKIP_THRESH_CM = 3.0  # ~half of 40 kHz skip distance (~4.3 cm)

mic1 = df[df['mic'] == 'Mic1'].copy()
mic1['is_skip'] = mic1['err_b_cal'].abs() > SKIP_THRESH_CM

print(f'\nOutlier rate (|err| > {SKIP_THRESH_CM} cm) by distance — Mic1 Path B:')
print(f'  {"dist_cm":>7}  {"n_trials":>9}  {"n_skips":>8}  {"skip_%":>8}  {"MAE_all":>8}  {"MAE_inlier":>11}  {"bias_inlier":>12}')
print('  ' + '-' * 75)
for d in sorted(mic1['distance_cm'].unique()):
    g = mic1[mic1['distance_cm'] == d]
    ns = g['is_skip'].sum()
    inlier = g[~g['is_skip']]
    print(f'  {int(d):>7}  {len(g):>9}  {ns:>8}  {ns/len(g)*100:>7.1f}%  '
          f'{g["err_b_cal"].abs().mean():>8.3f}  '
          f'{inlier["err_b_cal"].abs().mean():>11.3f}  '
          f'{inlier["err_b_cal"].mean():>12.3f}')

print(f'\nOutlier rate by Tx — Mic1 Path B (all distances):')
print(f'  {"tx":>5}  {"n_trials":>9}  {"n_skips":>8}  {"skip_%":>8}  {"mean_err_skip":>14}')
print('  ' + '-' * 60)
for tx in sorted(mic1['tx'].unique()):
    g = mic1[mic1['tx'] == tx]
    sk = g[g['is_skip']]
    mean_skip = sk['err_b_cal'].mean() if len(sk) else float('nan')
    print(f'  {tx:>5}  {len(g):>9}  {len(sk):>8}  {len(sk)/len(g)*100:>7.1f}%  {mean_skip:>14.2f}')

print('\nError distribution percentiles — Mic1 Path B:')
print(f'  {"dist_cm":>7}  {"p10":>6}  {"p25":>6}  {"p50":>6}  {"p75":>6}  {"p90":>6}  {"p99":>6}')
print('  ' + '-' * 55)
for d in sorted(mic1['distance_cm'].unique()):
    g = mic1[mic1['distance_cm'] == d]['err_b_cal']
    p = np.percentile(g, [10, 25, 50, 75, 90, 99])
    print(f'  {int(d):>7}  {p[0]:>6.2f}  {p[1]:>6.2f}  {p[2]:>6.2f}  {p[3]:>6.2f}  {p[4]:>6.2f}  {p[5]:>6.2f}')

inlier_mae = mic1[~mic1['is_skip']]['err_b_cal'].abs().mean()
all_mae = mic1['err_b_cal'].abs().mean()
skip_frac = mic1['is_skip'].mean()
print(f'\nHypothetical improvement if skips suppressed (Mic1 Path B):')
print(f'  All trials MAE:  {all_mae:.3f} cm  (skip rate {skip_frac*100:.1f}%)')
print(f'  Inlier-only MAE: {inlier_mae:.3f} cm')
print(f'  MAE reduction:   {all_mae - inlier_mae:.3f} cm')

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Angle-dependent bias on Mic2
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 90)
print('PART 2 — Angle-dependent envelope distortion on Mic2 (Path B, calibrated)')
print('=' * 90)

mic2 = df[df['mic'] == 'Mic2'].copy()

print(f'\nBias and MAE by angle — Mic2 Path B:')
print(f'  {"angle":>6}  {"n":>6}  {"bias_cm":>8}  {"MAE_cm":>8}  {"std_cm":>8}  {"p90_cm":>8}')
print('  ' + '-' * 60)
for ang in sorted(mic2['angle_deg'].unique()):
    g = mic2[mic2['angle_deg'] == ang]['err_b_cal']
    print(f'  {ang:>6}  {len(g):>6}  {g.mean():>8.3f}  {g.abs().mean():>8.3f}  {g.std():>8.3f}  {np.percentile(g.abs(), 90):>8.3f}')

print(f'\nBias by (angle, Tx) — Mic2 Path B:')
pivot_bias = mic2.groupby(['angle_deg', 'tx'])['err_b_cal'].mean().unstack()
print('  ' + pivot_bias.round(3).to_string())

print(f'\nMAE by (angle, Tx) — Mic2 Path B:')
pivot_mae = mic2.groupby(['angle_deg', 'tx'])['err_b_cal'].apply(lambda x: x.abs().mean()).unstack()
print('  ' + pivot_mae.round(3).to_string())

print(f'\nBias by (angle, distance) — Mic2 Path B:')
pivot_d = mic2.groupby(['angle_deg', 'distance_cm'])['err_b_cal'].mean().unstack()
print('  ' + pivot_d.round(3).to_string())

bias_0   = mic2[mic2['angle_deg'] == 0]['err_b_cal'].mean()
bias_p10 = mic2[mic2['angle_deg'] == 10]['err_b_cal'].mean()
bias_m10 = mic2[mic2['angle_deg'] == -10]['err_b_cal'].mean()
print(f'\n  Bias at   0°: {bias_0*10:.1f} mm')
print(f'  Bias at +10°: {bias_p10*10:.1f} mm  (Δ = {(bias_p10-bias_0)*10:+.1f} mm vs 0°)')
print(f'  Bias at -10°: {bias_m10*10:.1f} mm  (Δ = {(bias_m10-bias_0)*10:+.1f} mm vs 0°)')
print('  2. Transducer phase center offset from mechanical face')
print('\nIf residual bias scales with distance, likely causes:')
print('  1. Sound speed error (temperature mismatch)')
print('  2. Depth offset in reflector model')
