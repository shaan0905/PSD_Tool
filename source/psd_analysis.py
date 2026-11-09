import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os


# 1. Define the true Rosin-Rammler equation: R(d) = exp(-(d/d')^n)
def rr_func(d, d_prime, n):
    """
    Rosin-Rammler cumulative retained fraction.
    d: sieve size
    d_prime: characteristic particle size (63.2% passing)
    n: spread parameter
    """
    return np.exp(- (d / d_prime)**n)

def run_psd_calculations(df, sieve_sizes):
    """
    Dynamically calculates Cumulative % Passing and fits the Rosin-Rammler 
    distribution using Non-Linear Optimization (Curve Fitting).
    """
    results = []
    
    # Ensure sieve sizes are a numpy array and filter out the pan (0) for the fit
    sieves_array = np.array(sieve_sizes)
    valid_idx = sieves_array > 0
    d_vals = sieves_array[valid_idx]
    
    for index, row in df.iterrows():
        sample_name = row['Sample']
        category = row['Category']
        
        # 1. Basic Arithmetic
        masses = pd.to_numeric(row[sieve_sizes], errors='coerce').fillna(0).values
        total_mass = np.sum(masses)
        
        # Guard clause: skip completely empty samples
        if total_mass == 0:
            results.append({
                'Sample': sample_name,
                'Category': category,
                'Total Mass (g)': 0,
                "d' (um)": np.nan,
                'n': np.nan,
                'R2': np.nan
            })
            continue
            
        # Calculate Cumulative Retained Fraction (R_d)
        cum_retained = np.cumsum(masses) / total_mass
        r_vals = cum_retained[valid_idx] # Match the valid sieves
        
        # 2. Non-Linear Curve Fitting
        try:
            # We use curve_fit with an initial guess (p0=[1000, 1.0]) 
            # bounds=([0, 0], [infinity, infinity]) ensures d' and n are never negative
            popt, pcov = curve_fit(
                rr_func, 
                d_vals, 
                r_vals, 
                p0=[1000, 1.0], 
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=5000
            )
            d_prime = popt[0]
            n_param = popt[1]
            
            # Calculate R-squared for the non-linear fit
            residuals = r_vals - rr_func(d_vals, d_prime, n_param)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((r_vals - np.mean(r_vals))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
        except Exception:
            # If the curve fails to converge (e.g., weird or flat data)
            d_prime, n_param, r_squared = np.nan, np.nan, np.nan
            
        # 3. Store the outputs safely
        results.append({
            'Sample': sample_name,
            'Category': category,
            'Total Mass (g)': round(total_mass, 2),
            "d' (um)": round(d_prime, 2) if pd.notnull(d_prime) else np.nan,
            'n': round(n_param, 4) if pd.notnull(n_param) else np.nan,
            'R2': round(r_squared, 5) if pd.notnull(r_squared) else np.nan
        })
        
    return pd.DataFrame(results)


# ==========================================
# VERIFICATION TEST (Runs if executed directly)
# ==========================================
if __name__ == "__main__":
    # Dynamically locate the data file relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', '220125_Particle_Size_Distributions_EL.xlsx')
    
    try:
        # Import the ingestion script
        from preprocessing import extract_data
        
        # 1. Ingest
        clean_df, sizes = extract_data(file_path, sheet_name="Pre-milled")
        
        # 2. Calculate using the NON-LINEAR method
        analysis_df = run_psd_calculations(clean_df, sizes)
        
        print(f"--- SUCCESSFULLY PROCESSED {len(analysis_df)} SAMPLES (NON-LINEAR) ---")
        print("\nPreview of top 10 Rosin-Rammler outputs:")
        
        # Print the first 10 rows cleanly to verify
        print(analysis_df.head(10).to_string(index=False))
        
    except FileNotFoundError:
        print("Error: Could not find the Excel file or preprocessing.py.")
        print("Ensure you are running this from within the PSD-Tool/src/ folder.")