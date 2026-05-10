from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from TOF_estimation import PCB, SOUND_SPEED_M_S, tof_to_distance_m

csv_path = Path('outputs/tof_matched/multifrequenz_denoised_branchA_sym6.csv')
df = pd.read_csv(csv_path)

methods = {'Path A (BP)': 'tof_path_a_s', 'Path B': 'tof_path_b_s'}

def apply_tau(frame, tof_col, tau_s):
    out = frame.copy()
    out['est_dist_cal_cm'] = [
        100.0 * tof_to_distance_m(getattr(r, tof_col) - tau_s, r.mic, r.tx, PCB, SOUND_SPEED_M_S)
        for r in frame.itertuples(index=False)
    ]
    out['err'] = out['est_dist_cal_cm'] - out['distance_cm']
    return out

def fit_objective_median(frame, tof_col, tau_s):
    """Median-based objective: more robust to cycle-skip outliers."""
    import numpy as np
    return np.median(apply_tau(frame, tof_col, tau_s)['err'])

print('=' * 100)
print('STEP 1: Fit delay per (Tx, Mic, Method) using ALL mics')
print('=' * 100)

global_calibration = []
for method_name, tof_col in methods.items():
    print(f'\n{method_name}')
    print('-' * 100)
    print(f'{"Tx":>5} {"Mic":>5} {"tau_us":>10} {"MAE_cm":>8} {"std_cm":>8}')
    print('-' * 100)
    
    for tx_name in sorted(df['tx'].unique()):
        for mic_name in ['Mic1', 'Mic2', 'Mic3']:
            subset = df[(df['tx'] == tx_name) & (df['mic'] == mic_name)]
            
            tau_s = brentq(
                lambda tau: fit_objective_median(subset, tof_col, tau),
                -200e-6, 200e-6,
            )
            result = apply_tau(subset, tof_col, tau_s)
            
            global_calibration.append({
                'method': method_name,
                'tx': tx_name,
                'mic': mic_name,
                'tau_s': tau_s,
                'tau_us': tau_s * 1e6,
                'mae': result['err'].abs().mean(),
                'std': result['err'].std(),
            })
            
            print(f'{tx_name:>5} {mic_name:>5} {tau_s*1e6:>10.3f} {result["err"].abs().mean():>8.3f} {result["err"].std():>8.3f}')

global_cal_df = pd.DataFrame(global_calibration)

# Evaluate on full dataset with these calibrations
print('\n' + '=' * 100)
print('STEP 2: Apply calibration to full dataset and evaluate')
print('=' * 100)

for method_name, tof_col in methods.items():
    print(f'\n{method_name}')
    print('=' * 100)
    
    cal_subset = global_cal_df[global_cal_df['method'] == method_name]
    tau_dict = {(r['tx'], r['mic']): r['tau_s'] for _, r in cal_subset.iterrows()}
    
    df_copy = df.copy()
    df_copy['est_cal_cm'] = [
        100.0 * tof_to_distance_m(
            getattr(r, tof_col) - tau_dict.get((r.tx, r.mic), 0.0),
            r.mic, r.tx, PCB, SOUND_SPEED_M_S,
        )
        for r in df.itertuples(index=False)
    ]
    df_copy['err_cal_cm'] = df_copy['est_cal_cm'] - df_copy['distance_cm']
    
    if 'Path B' in method_name:
        df_copy['err_uncal_cm'] = df_copy['est_dist_path_b_cm'] - df_copy['distance_cm']
    else:
        df_copy['err_uncal_cm'] = df_copy['est_dist_path_a_cm'] - df_copy['distance_cm']
    
    # Overall stats
    mae_cal_all = df_copy['err_cal_cm'].abs().mean()
    mae_uncal_all = df_copy['err_uncal_cm'].abs().mean()
    
    print(f'\nFULL DATASET (all mics):')
    print(f'  Uncalibrated: MAE={mae_uncal_all:.3f} cm')
    print(f'  Calibrated:   MAE={mae_cal_all:.3f} cm  improvement={mae_uncal_all-mae_cal_all:+.3f} cm')
    
    # Per-mic breakdown
    for mic_name in ['Mic1', 'Mic2', 'Mic3']:
        df_mic = df_copy[df_copy['mic'] == mic_name]
        mae_cal = df_mic['err_cal_cm'].abs().mean()
        mae_uncal = df_mic['err_uncal_cm'].abs().mean()
        print(f'\n{mic_name}:')
        print(f'  Uncalibrated: MAE={mae_uncal:.3f} cm')
        print(f'  Calibrated:   MAE={mae_cal:.3f} cm  improvement={mae_uncal-mae_cal:+.3f} cm')
    
    # Per-distance breakdown for full dataset
    print(f'\nPER-DISTANCE (full dataset):')
    print(f'{"distance":>8} {"uncal_mae":>10} {"cal_mae":>8} {"cal_bias":>9} {"≤2cm%":>8} {"Mic1+2":>8} {"Mic3":>8}')
    print('-' * 85)
    for distance_cm in sorted(df_copy['distance_cm'].unique()):
        g_all = df_copy[df_copy['distance_cm'] == distance_cm]
        g_12 = g_all[g_all['mic'].isin(['Mic1', 'Mic2'])]
        g_3 = g_all[g_all['mic'] == 'Mic3']
        
        mae_u = g_all['err_uncal_cm'].abs().mean()
        mae_c = g_all['err_cal_cm'].abs().mean()
        bias = g_all['err_cal_cm'].mean()
        pct = (g_all['err_cal_cm'].abs() <= 2.0).mean() * 100.0
        mae_12 = g_12['err_cal_cm'].abs().mean() if len(g_12) > 0 else 0
        mae_3 = g_3['err_cal_cm'].abs().mean() if len(g_3) > 0 else 0
        
        print(f'{int(distance_cm):>7}  {mae_u:>9.3f}  {mae_c:>7.3f}  {bias:>8.3f}  {pct:>7.1f}%  {mae_12:>7.3f}  {mae_3:>7.3f}')

# Save calibrated CSV
print('\n' + '=' * 100)
print('STEP 3: Save calibrated CSV')
print('=' * 100)

output_csv = Path('outputs/tof_matched/multifrequenz_denoised_branchA_sym6_cal_all.csv')

for method_name, tof_col in methods.items():
    cal_subset = global_cal_df[global_cal_df['method'] == method_name]
    tau_dict = {(r['tx'], r['mic']): r['tau_s'] for _, r in cal_subset.iterrows()}
    
    col_name = f'est_dist_{"path_a_bp" if "Path A" in method_name else "path_b"}_cal_cm'
    df[col_name] = [
        100.0 * tof_to_distance_m(
            getattr(r, tof_col) - tau_dict.get((r.tx, r.mic), 0.0),
            r.mic, r.tx, PCB, SOUND_SPEED_M_S,
        )
        for r in df.itertuples(index=False)
    ]

df.to_csv(output_csv, index=False)
print(f'Saved to {output_csv}')
print(f'New columns: est_dist_path_a_bp_cal_cm, est_dist_path_b_cal_cm')
