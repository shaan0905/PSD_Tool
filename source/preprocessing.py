import pandas as pd
import numpy as np
import os


def extract_data(filepath: str, sheet_name: str = "Pre-milled") -> pd.DataFrame:
    """
    Extracts raw sieve mass data from either 'Pre-milled' or 'Milled' sheets.
    Dynamically tracks section categories and varying sieve aperture stacks.
    Returns a unified long/tidy DataFrame ready for distribution processing.
    """
    df_raw = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
    
    # Locate the header row containing 'Sample' in column A
    sample_header_idx = None
    for idx, val in enumerate(df_raw.iloc[:, 0]):
        if str(val).strip().lower() == 'sample':
            sample_header_idx = idx
            break
            
    if sample_header_idx is None:
        raise ValueError(f"Could not locate 'Sample' header in sheet '{sheet_name}'.")

    # Initial sieve apertures from row 2
    raw_apertures = df_raw.iloc[sample_header_idx, 1:10].values
    current_apertures = [
        0.0 if str(s).strip().lower() == 'pan' 
        else (float(s) * 1000.0 if float(s) < 50.0 else float(s))
        for s in raw_apertures
    ]

    current_category = 'Pure Samples'
    records = []

    # Iterate over rows below the main header
    for r in range(sample_header_idx + 1, len(df_raw)):
        sample_name = df_raw.iloc[r, 0]
        if pd.isna(sample_name) or str(sample_name).strip() == '':
            continue
            
        row_vals = df_raw.iloc[r, 1:10].values
        
        # Check if this row is a Category Header / Sieve Reset (e.g. '10% PKS')
        is_all_na = pd.isna(row_vals).all()
        has_pan = any(str(v).strip().lower() == 'pan' for v in row_vals if pd.notna(v))
        
        if is_all_na or has_pan:
            current_category = str(sample_name).strip()
            # If the header row defines new sieve apertures (like row 15 in Milled)
            if has_pan:
                current_apertures = [
                    0.0 if str(s).strip().lower() == 'pan' 
                    else (float(s) * 1000.0 if float(s) < 50.0 else float(s))
                    for s in row_vals
                ]
            continue
            
        # Parse numeric masses
        masses = [pd.to_numeric(v, errors='coerce') for v in row_vals]
        if all(pd.isna(m) for m in masses):
            continue
        masses = [0.0 if pd.isna(m) else float(m) for m in masses]
        
        # Record sample row with its corresponding sieve stack
        for aperture, mass in zip(current_apertures, masses):
            records.append({
                'Sample': str(sample_name).strip(),
                'Category': current_category,
                'Sieve_um': aperture,
                'Mass_g': mass
            })

    return pd.DataFrame(records)


def calculate_distributions(df_long: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Computes total recovered sample mass, percentage mass retained, and
    cumulative passing (undersize) for each individual sample, respecting
    its specific sieve stack apertures.
    """
    processed_samples = []
    
    for (sample, category), group in df_long.groupby(['Sample', 'Category'], sort=False):
        # Sort sieves from coarsest down to pan (0 um)
        grp = group.sort_values(by='Sieve_um', ascending=False).copy()
        
        total_mass = grp['Mass_g'].sum()
        if total_mass <= 0:
            continue
            
        grp['Pct_Mass'] = (grp['Mass_g'] / total_mass) * 100.0
        grp['Cum_Retained'] = grp['Pct_Mass'].cumsum()
        grp['Cum_Passing'] = np.clip(100.0 - grp['Cum_Retained'], 0.0, 100.0)
        
        # Determine appropriate top boundary: 6000 um for 4750 um stacks, 4750 um for 3350 um stacks
        max_sieve = grp['Sieve_um'].max()
        top_bound = 6000.0 if max_sieve > 4000.0 else 4750.0
        
        sieve_array = np.array([top_bound] + [10.0 if s == 0.0 else s for s in grp['Sieve_um'].values])
        cum_array = np.array([100.0] + list(grp['Cum_Passing'].values))
        cum_array[-1] = 0.0  # Force exact 0.0% for the pan / 10 um bound
        
        processed_samples.append({
            'Sample': sample,
            'Category': category,
            'Total_Mass_g': total_mass,
            'Sieves_um': sieve_array,
            'Cum_Passing': cum_array
        })

    return df_long, processed_samples