import os
import numpy as np
import pandas as pd


def _parse_aperture_um(val) -> float:
    """Converts raw header values (mm, um, or 'pan') into micrometres (um)."""
    s_str = str(val).strip().lower()
    if s_str == 'pan':
        return 0.0
    num = float(val)
    # Values < 50 are in mm (e.g. 3.35 mm -> 3350 um), unless 0
    return num * 1000.0 if (0.0 < num < 50.0) else num


def _find_sieve_end_col(row_values) -> int:
    """
    Finds the column index of 'pan' (inclusive) across a header row so the
    Excel sheet can have any number of sieve columns.
    """
    for idx in range(1, len(row_values)):
        val = row_values[idx]
        if pd.isna(val):
            break
        s_str = str(val).strip().lower()
        if s_str == 'pan':
            return idx + 1
        if s_str == 'total':
            return idx
    return 10  # Fallback to standard 9 sieve columns (cols 1..9)


def extract_data(filepath: str, sheet_name: str = "Pre-milled") -> pd.DataFrame:
    """
    Extracts raw sieve mass data from Excel sheets with any number of sieve columns.
    Dynamically tracks section categories and mid-sheet sieve stack changes.
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

    header_row = df_raw.iloc[sample_header_idx].values
    end_col = _find_sieve_end_col(header_row)
    raw_apertures = header_row[1:end_col]
    current_apertures = [_parse_aperture_um(s) for s in raw_apertures]

    current_category = 'Pure Samples'
    records = []

    # Iterate over rows below the main header
    for r in range(sample_header_idx + 1, len(df_raw)):
        sample_name = df_raw.iloc[r, 0]
        if pd.isna(sample_name) or str(sample_name).strip() == '':
            continue
            
        full_row = df_raw.iloc[r].values
        row_vals = full_row[1:end_col]
        
        # Check if this row is a Category Header or a Mid-Sheet Sieve Header Reset
        is_all_na = pd.isna(row_vals).all()
        has_pan = any(str(v).strip().lower() == 'pan' for v in full_row[1:20] if pd.notna(v))
        
        if is_all_na or has_pan:
            current_category = str(sample_name).strip()
            if has_pan:
                end_col = _find_sieve_end_col(full_row)
                current_apertures = [_parse_aperture_um(s) for s in full_row[1:end_col]]
            continue
            
        # Parse numeric masses
        masses = [pd.to_numeric(v, errors='coerce') for v in row_vals]
        if all(pd.isna(m) for m in masses):
            continue
        masses = [0.0 if pd.isna(m) else float(m) for m in masses]
        
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
    cumulative passing (undersize) for each sample across any custom sieve stack.
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
        
        # Determine 100% passing upper boundary (preserve 4750/6000 for standard stacks, or scale by 1.4x for custom stacks)
        max_sieve = grp['Sieve_um'].max()
        if np.isclose(max_sieve, 3350.0):
            top_bound = 4750.0
        elif np.isclose(max_sieve, 4750.0):
            top_bound = 6000.0
        else:
            top_bound = round(max_sieve * 1.4, 1)
            
        sieve_array = np.array([top_bound] + [10.0 if s == 0.0 else s for s in grp['Sieve_um'].values])
        cum_array = np.array([100.0] + list(grp['Cum_Passing'].values))
        cum_array[-1] = 0.0  # Force exact 0.0% on the pan
        
        processed_samples.append({
            'Sample': sample,
            'Category': category,
            'Total_Mass_g': total_mass,
            'Sieves_um': sieve_array,
            'Cum_Passing': cum_array
        })

    return df_long, processed_samples