import os
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit

from preprocessing import extract_data, calculate_distributions


def rosin_rammler_undersize(d: np.ndarray, d_prime: float, n: float) -> np.ndarray:
    """
    Rosin-Rammler cumulative undersize (% passing) model:
    Q(d) = 100 * (1 - exp(-(d / d')^n))
    """
    return 100.0 * (1.0 - np.exp(-((d / d_prime) ** n)))


def fit_rosin_rammler(sieve_um: np.ndarray, cum_passing: np.ndarray) -> tuple[float, float, float, float]:
    """
    Fits Rosin-Rammler parameters (d', n) using non-linear least squares.
    Returns: (d_prime, n, r_squared, pearson_r)
    """
    # Use PCHIP d50 as an initial guess for d'
    x_inc, idx = np.unique(cum_passing[::-1], return_index=True)
    y_inc = sieve_um[::-1][idx]
    d50_guess = float(np.interp(50.0, x_inc, y_inc))
    
    popt, _ = curve_fit(
        rosin_rammler_undersize,
        sieve_um,
        cum_passing,
        p0=[max(d50_guess, 100.0), 1.2],
        bounds=([1.0, 0.05], [50000.0, 25.0]),
        maxfev=20000
    )
    d_prime, n = float(popt[0]), float(popt[1])
    
    # Goodness-of-fit metrics
    y_pred = rosin_rammler_undersize(sieve_um, d_prime, n)
    ss_res = np.sum((cum_passing - y_pred) ** 2)
    ss_tot = np.sum((cum_passing - np.mean(cum_passing)) ** 2)
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # Pearson r (Excel R2 column)
    if np.std(cum_passing) > 0 and np.std(y_pred) > 0:
        pearson_r = float(np.corrcoef(cum_passing, y_pred)[0, 1])
    else:
        pearson_r = 0.0
        
    return d_prime, n, r_squared, pearson_r


def interpolate_percentiles_pchip(
    sieve_um: np.ndarray, 
    cum_passing: np.ndarray, 
    percentiles: list[int] = [10, 50, 5, 16, 25, 30, 60, 75, 80, 84, 90, 95]
) -> dict[str, float]:
    """
    Calculates d_x particle sizes (in um) using monotonic PCHIP interpolation
    to match the values in 'Analysis Pre-milled' and 'Analysis Milled'.
    """
    q_rev = cum_passing[::-1]
    d_rev = sieve_um[::-1]
    
    # Ensure strictly increasing x coordinates for PCHIP
    q_unique, unique_indices = np.unique(q_rev, return_index=True)
    d_unique = d_rev[unique_indices]
    
    pchip = PchipInterpolator(q_unique, d_unique)
    interp_vals = pchip(percentiles)
    
    return {f"d{p}": float(val) for p, val in zip(percentiles, interp_vals)}


def run_psd_analysis_dataset(processed_samples: list[dict]) -> pd.DataFrame:
    """
    Runs Rosin-Rammler fits (filling all blank blend rows), PCHIP percentiles,
    and dispersion/shape indices across every sample in the dataset.
    """
    target_percentiles = [10, 50, 5, 16, 25, 30, 60, 75, 80, 84, 90, 95]
    results = []

    for item in processed_samples:
        s_um = item['Sieves_um']
        cum_vals = item['Cum_Passing']
        
        # 1. Non-linear Rosin-Rammler Curve Fit
        d_prime, n, r2, pearson_r = fit_rosin_rammler(s_um, cum_vals)
        
        # 2. PCHIP Percentile Interpolations
        d_vals = interpolate_percentiles_pchip(s_um, cum_vals, target_percentiles)
        
        d5, d10, d16, d25 = d_vals['d5'], d_vals['d10'], d_vals['d16'], d_vals['d25']
        d30, d50, d60, d75 = d_vals['d30'], d_vals['d50'], d_vals['d60'], d_vals['d75']
        d80, d84, d90, d95 = d_vals['d80'], d_vals['d84'], d_vals['d90'], d_vals['d95']
        
        # 3. Dispersion Metrics
        hs = d84 / d50
        ls = d50 / d16
        mrs = (d90 - d10) / d50
        cu = d60 / d10
        cc = (d30 ** 2) / (d60 * d10)
        
        # 4. Shape Metrics
        gsi = ((d16 + d84 - 2.0 * d50) / (2.0 * (d84 - d16))) + (
            (d5 + d95 - 2.0 * d50) / (2.0 * (d95 - d5))
        )
        kg = (d95 - d5) / (2.44 * (d75 - d25))
        
        results.append({
            'Sample': item['Sample'],
            'Category': item['Category'],
            'Total_Mass_g': item['Total_Mass_g'],
            "d' (um)": d_prime,
            'n': n,
            'R2 (Excel r)': pearson_r,
            'R2 (Standard)': r2,
            **d_vals,
            'hs': hs,
            'ls': ls,
            'MRS': mrs,
            'Cu': cu,
            'Cc': cc,
            'Gsi': gsi,
            'Kg': kg
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', '220125_Particle_Size_Distributions_EL.xlsx')

    for sheet in ["Pre-milled", "Milled"]:
        print(f"\n{'='*30} Processing {sheet} {'='*30}")
        df_raw_long = extract_data(file_path, sheet_name=sheet)
        _, processed_samples = calculate_distributions(df_raw_long)
        df_analysis = run_psd_analysis_dataset(processed_samples)
        
        print(f"Successfully processed {len(df_analysis)} samples for {sheet}.")
        print(df_analysis[['Sample', 'Category', "d' (um)", 'n', 'd10', 'd50', 'd90', 'MRS', 'Cu']].head(10).round(2))