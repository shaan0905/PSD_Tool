import pandas as pd
import numpy as np

def extract_data(filepath, sheet_name="Pre-milled"):
    """
    Extracts data from sample excel sheet, this is just for the pre milled page for now, but I can update this for the post millage data too.
    """
    # Read the Excel file 
    df_raw = pd.read_excel(filepath, sheet_name=sheet_name, header=2)
    
    # Extract the samples from this specific excel sheet, just the pre milled and currently won't work for additional data sheets,
    # Will have to make some sort of decision on how imputs will work, excel sheet or csv file, or a straight input of the data. 
    # My current best idea is to allow the user to input each row of data, and then compile all of these entries before running the calculation.
    df = df_raw.iloc[:, 0:10].copy()
    
    # Standardize sieve column names
    # Replace 'pan' or 'PAN' with 0
    df.columns = [0 if str(c).lower() == 'pan' else c for c in df.columns]
    
    # Convert mm to um (if a sieve size is less than 50, it is in mm), since variance of units found in the excel sheet. 
    # This conversion could be useful for the final tool, as it will allow for a more standardized input of data. Maybe unit choice prompts?
    raw_sizes = df.columns[1:].values
    sieve_sizes_um = [
        float(s) * 1000 if isinstance(s, (int, float)) and float(s) < 50 else float(s) 
        for s in raw_sizes
    ]
    df.columns = ['Sample'] + sieve_sizes_um
    
    # Clean empty rows
    df = df.dropna(subset=['Sample']) # Drop rows where Column A is completely empty
    
    # Tag Categories
    # Filter for categories, where the first 9 columns are all NaN, and the first column is not NaN.
    df['Category'] = np.where(df.iloc[:, 1:10].isna().all(axis=1), df['Sample'], np.nan)
    
    # Forward-fill the category down the rows. (Default to 'Pure Samples' for the top section)
    df['Category'] = df['Category'].ffill().fillna('Pure Samples')
    
    # Drop the section header rows themselves now that we have the tags
    df = df.dropna(subset=sieve_sizes_um, how='all').reset_index(drop=True)
    
    # Ensure all mass data is numeric (fill any missing/blank cells with 0)
    df[sieve_sizes_um] = df[sieve_sizes_um].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Reorder columns to put Category next to Sample
    cols = ['Sample', 'Category'] + sieve_sizes_um
    df = df[cols]
    
    return df, sieve_sizes_um
import os



def calculate_distributions(
    clean_df: pd.DataFrame, 
    sieve_sizes_um: list[float], 
    upper_bound_um: float = 4750.0,
    pan_cum_label: float = 10.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Takes the cleaned wide-format DataFrame from extract_data() and calculates:
    1. df_pct: Total mass and Percentage Mass retained per sieve (row-wise).
    2. df_cum: Cumulative Percentage Passing (Undersize) matching the Excel sheet.
    """
    meta_cols = ['Sample', 'Category']
    masses = clean_df[sieve_sizes_um].copy()
    
    # 1. Row-wise total mass (g) across all sieves + pan
    total_mass = masses.sum(axis=1)
    if (total_mass <= 0).any():
        raise ValueError("One or more samples have a total mass <= 0. Check input data.")
        
    # 2. Percentage mass retained on each sieve (%)
    pct_mass = masses.div(total_mass, axis=0) * 100.0
    
    # Assemble Percentage Mass DataFrame
    df_pct = pd.concat(
        [clean_df[meta_cols], total_mass.rename('Total_Mass_g'), pct_mass], 
        axis=1
    )
    
    # 3. Cumulative Distribution (% Passing / Undersize)
    # Running sum of % mass retained from coarsest sieve (3350 um) to pan (0 um)
    cum_retained = pct_mass.cumsum(axis=1)
    
    # Subtract cumulative retained from 100% (clipping floating-point drift like -1.26e-14 to 0.0)
    cum_passing = np.clip(100.0 - cum_retained, 0.0, 100.0)
    cum_passing.loc[:, 0.0] = cum_passing.loc[:, 0.0].round(10) # Force exact 0.0 on the pan
    
    # Assemble Cumulative Distribution DataFrame with the 4750 um (100%) starting column
    df_cum = clean_df[meta_cols].copy()
    df_cum[upper_bound_um] = 100.0
    
    for size in sieve_sizes_um:
        # Match Excel's convention where the pan end-point is labelled 10.0 um in the cumulative table
        col_name = pan_cum_label if size == 0.0 else size
        df_cum[col_name] = cum_passing[size]
        
    return df_pct, df_cum
 
if __name__ == "__main__":
    # Get the directory where ingestion.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the root, then into the data folder
    file_path = os.path.join(current_dir, '..', 'data', '220125_Particle_Size_Distributions_EL.xlsx')
    
    # Run the extraction
    clean_df, sizes = extract_data(file_path)
    print("Successfully Extracted Data:")
    print(clean_df)
    print(f"Total rows extracted: {len(clean_df)}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', '220125_Particle_Size_Distributions_EL.xlsx')
    
    # 1. Run the raw mass extraction
    clean_df, sizes = extract_data(file_path, sheet_name="Pre-milled")
    
    # 2. Compute Percentage Mass and Cumulative Distributions
    df_pct, df_cum = calculate_distributions(clean_df, sizes)
    
    print("--- Percentage Mass Retained (%) ---")
    print(df_pct.head().to_string(index=False))
    
    print("\n--- Cumulative Distribution (% Passing) ---")
    print(df_cum.head().to_string(index=False))