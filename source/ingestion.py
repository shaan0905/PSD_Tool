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