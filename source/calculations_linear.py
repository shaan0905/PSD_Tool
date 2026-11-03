import numpy as np
import pandas as pd
from scipy.stats import linregress

def run_psd_calculations(df, sieve_sizes):
    """
    Dynamically calculates Cumulative % Passing and fits the Rosin-Rammler distribution.
    Scales automatically to any number of samples and any stack of sieve sizes.
    """
    results = []
    
    for index, row in df.iterrows():
        sample_name = row['Sample']
        category = row['Category']
        
        # Dynamic Extraction 
        # Grabs the masses using whatever sieve sizes were passed to the function
        masses = pd.to_numeric(row[sieve_sizes], errors='coerce').fillna(0).values
        total_mass = np.sum(masses)
        
        # Guard clause: skip completely empty samples to prevent divide-by-zero
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
            
        # Basic Sieve Arithmetic
        percent_retained = (masses / total_mass) * 100
        cum_retained = np.cumsum(percent_retained)
        # (Optional: cum_passing = 100.0 - cum_retained if you need it for plotting later)
        
        # Rosin-Rammler Linearized Regression
        x_fit = []
        y_fit = []
        
        for i, size in enumerate(sieve_sizes):
            # Ignore the pan (size 0) because log(0) is mathematically undefined
            if size > 0:
                R_d = cum_retained[i] / 100.0
                
                # The double-log equation only works for fractions strictly between 0 and 1
                if 0 < R_d < 1.0:
                    x_fit.append(np.log(size))
                    y_fit.append(np.log(-np.log(R_d)))
                    
        # Perform the regression (requires at least 2 valid points to draw a line)
        if len(x_fit) > 1:
            try:
                slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit)
                n_param = slope
                d_prime = np.exp(-intercept / n_param)
                r_squared = r_value ** 2
            except Exception:
                # Catch any bizarre mathematical anomalies (e.g. infinite slopes)
                n_param, d_prime, r_squared = np.nan, np.nan, np.nan
        else:
            n_param, d_prime, r_squared = np.nan, np.nan, np.nan
            
        # Store the outputs cleanly
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
    import os
    import sys
    
    # Dynamically locate the data file relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', '220125_Particle_Size_Distributions_EL.xlsx')
    
    # Import the ingestion script you just perfected
    try:
        from ingestion import extract_data
        
        # Ingest
        clean_df, sizes = extract_data(file_path, sheet_name="Pre-milled")
        
        # Calculate
        analysis_df = run_psd_calculations(clean_df, sizes)
        
        print(f"--- SUCCESSFULLY PROCESSED {len(analysis_df)} SAMPLES ---")
        print("\nPreview of top 10 Rosin-Rammler outputs:")
        
        # Print the first 10 rows cleanly to verify 
        print(analysis_df.head(10).to_string(index=False))
        
    except FileNotFoundError:
        print("Error: Could not find the Excel file or ingestion.py.")
        