import io
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# python -m streamlit run app/main_app.py
# Allow importing backend modules from the ../source directory
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, '..', 'source')
if source_dir not in sys.path:
    sys.path.append(source_dir)

from preprocessing import extract_data, calculate_distributions
from psd_analysis import run_psd_analysis_dataset, rosin_rammler_undersize
from plotting import setup_plot_style, plot_psd_curve, finalize_axis

# --- Page Configuration ---
st.set_page_config(
    page_title="PSD-Tool | Biomass & Coal Particle Size Analyser",
    layout="wide"
)

# ==============================================================================
# Helper Functions for Templates & Wide <-> Long Data Conversion
# ==============================================================================

def parse_custom_sieve_string(sieve_str: str) -> list[float]:
    """Parses a comma-separated string of sieve sizes (um) into a sorted descending list."""
    sizes = []
    for token in sieve_str.split(','):
        token = token.strip()
        if not token or token.lower() == 'pan':
            continue
        try:
            val = float(token)
            if val > 0:
                sizes.append(val)
        except ValueError:
            continue
    return sorted(list(set(sizes)), reverse=True)


def generate_excel_template(custom_sieves_um: list[float]) -> bytes:
    """
    Generates a clean Excel (.xlsx) template in memory using the user's
    selected/custom sieve sizes (supports any number of sieves).
    """
    output = io.BytesIO()
    num_cols = 1 + len(custom_sieves_um) + 1  # Sample + sieves + pan
    
    # Generate realistic mock masses matching the number of custom sieves
    rng = np.random.default_rng(42)
    ex1 = np.round(rng.uniform(5.0, 65.0, size=len(custom_sieves_um) + 1), 1).tolist()
    ex2 = np.round(rng.uniform(2.0, 80.0, size=len(custom_sieves_um) + 1), 1).tolist()
    ex3 = np.round(rng.uniform(4.0, 55.0, size=len(custom_sieves_um) + 1), 1).tolist()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name in ["Pre-milled", "Milled"]:
            rows = [
                [sheet_name, "Weighed mass (g)"] + [None] * (num_cols - 2),
                [None] * num_cols,
                ["Sample"] + custom_sieves_um + ["pan"],
                ["Example_Sample_1"] + ex1,
                ["Example_Sample_2"] + ex2,
                [None] * num_cols,
                ["10% blends PKS"] + [None] * (num_cols - 1),
                [None] * num_cols,
                ["Blend_Sample_1"] + ex3,
            ]
            df_template = pd.DataFrame(rows)
            df_template.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            
    return output.getvalue()


def long_to_wide_df(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the backend long DataFrame into a wide, editable table
    where each row is a Sample and columns are Sieve apertures (um).
    """
    if df_long.empty:
        cols = ['Sample', 'Category', '3350', '2360', '1000', '850', '600', '300', '212', '106', 'pan (0 um)']
        return pd.DataFrame(columns=cols)

    df_wide = df_long.pivot_table(
        index=['Sample', 'Category'],
        columns='Sieve_um',
        values='Mass_g',
        aggfunc='first',
        sort=False
    ).reset_index().fillna(0.0)

    # Sort sieve columns from coarsest down to pan (0.0)
    meta_cols = ['Sample', 'Category']
    sieve_cols = sorted([c for c in df_wide.columns if c not in meta_cols], reverse=True)
    df_wide = df_wide[meta_cols + sieve_cols]

    rename_map = {c: ("pan (0 um)" if c == 0.0 else f"{float(c):g}") for c in sieve_cols}
    return df_wide.rename(columns=rename_map)


def wide_to_long_df(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the user-edited wide table from st.data_editor back into
    the long format required by calculate_distributions().
    """
    meta_cols = ['Sample', 'Category']
    sieve_str_cols = [c for c in df_wide.columns if c not in meta_cols]
    records = []

    for _, row in df_wide.iterrows():
        sample_name = str(row.get('Sample', '')).strip()
        if not sample_name or sample_name.lower() in ['nan', 'none']:
            continue
        category = str(row.get('Category', 'Pure Samples')).strip()
        if not category or category.lower() in ['nan', 'none']:
            category = 'Pure Samples'

        for col_str in sieve_str_cols:
            val = pd.to_numeric(row[col_str], errors='coerce')
            if pd.isna(val):
                continue
            aperture = 0.0 if 'pan' in str(col_str).lower() else float(col_str)
            records.append({
                'Sample': sample_name,
                'Category': category,
                'Sieve_um': aperture,
                'Mass_g': max(0.0, float(val))
            })

    return pd.DataFrame(records)


def insert_sieve_column_into_wide_df(df_wide: pd.DataFrame, new_sieve_um: float) -> pd.DataFrame:
    """Inserts a new sieve size column into the wide DataFrame in descending order."""
    col_label = f"{float(new_sieve_um):g}"
    if col_label in df_wide.columns:
        return df_wide

    df_updated = df_wide.copy()
    df_updated[col_label] = 0.0

    # Re-sort sieve columns from coarsest down to pan (0 um)
    meta_cols = ['Sample', 'Category']
    sieve_cols = [c for c in df_updated.columns if c not in meta_cols]
    
    def col_sort_key(c_name: str) -> float:
        return -1.0 if 'pan' in c_name.lower() else float(c_name)

    sorted_sieve_cols = sorted(sieve_cols, key=col_sort_key, reverse=True)
    return df_updated[meta_cols + sorted_sieve_cols]


# ==============================================================================
# Sidebar: Custom Sieve Stack, Template Download & Data Source Selection
# ==============================================================================

st.title("PSD-Tool: Particle Size Distribution & Rosin-Rammler Analyser")

st.sidebar.header("1. Sieve Stack & Excel Template")
sieve_preset = st.sidebar.selectbox(
    "Sieve Stack Preset:",
    [
        "Standard Stack (8 sieves: 3350 - 106 um)",
        "Coarse Blend Stack (8 sieves: 4750 - 106 um)",
        "Extended Stack (12 sieves: 6700 - 38 um)",
        "Custom Sieve List"
    ]
)

if "Standard" in sieve_preset:
    default_sieve_str = "3350, 2360, 1000, 850, 600, 300, 212, 106"
elif "Coarse" in sieve_preset:
    default_sieve_str = "4750, 3350, 2360, 1018, 600, 300, 212, 106"
elif "Extended" in sieve_preset:
    default_sieve_str = "6700, 4750, 3350, 2360, 1700, 1000, 850, 600, 300, 212, 106, 63, 38"
else:
    default_sieve_str = "4000, 2000, 1000, 500, 250, 125, 63"

custom_sieve_input = st.sidebar.text_input(
    "Sieve Apertures in um (comma-separated, add/remove freely):",
    value=default_sieve_str,
    help="Type any sieve sizes in micrometres separated by commas. They will automatically sort from coarsest to finest, and 'pan' is included automatically."
)
active_template_sieves = parse_custom_sieve_string(custom_sieve_input)

st.sidebar.download_button(
    label=f"📥 Download .xlsx Template ({len(active_template_sieves)} Sieves)",
    data=generate_excel_template(active_template_sieves),
    file_name="PSD_Tool_Input_Template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)

st.sidebar.divider()
st.sidebar.header("2. Data Input Mode")
input_mode = st.sidebar.radio(
    "Choose how to start:",
    [
        "Upload Excel File (+ Inspect/Edit)",
        "Load Default Dataset (+ Inspect/Edit)",
        "Manual Entry From Scratch"
    ]
)

default_excel = os.path.join(current_dir, '..', 'data', '220125_Particle_Size_Distributions_EL.xlsx')

# Determine source signature so session state resets cleanly when switching files/sheets/presets
if input_mode == "Upload Excel File (+ Inspect/Edit)":
    uploaded_file = st.sidebar.file_uploader("Upload Sieve Data (.xlsx)", type=["xlsx"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name
        xls = pd.ExcelFile(temp_path)
        sheet_choice = st.sidebar.selectbox("Select Sheet", xls.sheet_names)
        source_sig = f"upload_{uploaded_file.name}_{sheet_choice}"
        if st.session_state.get("source_sig") != source_sig:
            df_initial_long = extract_data(temp_path, sheet_name=sheet_choice)
            st.session_state["working_wide_df"] = long_to_wide_df(df_initial_long)
            st.session_state["source_sig"] = source_sig
    else:
        st.info("👈 Upload an `.xlsx` file in the sidebar (you can add as many sieve columns before `pan` as you want!), or switch to **Manual Entry From Scratch**.")
        st.stop()

elif input_mode == "Load Default Dataset (+ Inspect/Edit)":
    sheet_choice = st.sidebar.selectbox("Select Sheet", ["Pre-milled", "Milled"])
    source_sig = f"default_{sheet_choice}"
    if st.session_state.get("source_sig") != source_sig:
        df_initial_long = extract_data(default_excel, sheet_name=sheet_choice)
        st.session_state["working_wide_df"] = long_to_wide_df(df_initial_long)
        st.session_state["source_sig"] = source_sig

else:  # Manual Entry From Scratch
    sheet_choice = "Manual_Custom"
    source_sig = f"manual_{custom_sieve_input}"
    if st.session_state.get("source_sig") != source_sig:
        sieve_headers = [f"{s:g}" for s in active_template_sieves] + ['pan (0 um)']
        starter_data = {'Sample': ['Sample_1'], 'Category': ['Pure Samples']}
        for h in sieve_headers:
            starter_data[h] = [0.0]
        st.session_state["working_wide_df"] = pd.DataFrame(starter_data)
        st.session_state["source_sig"] = source_sig


# ==============================================================================
# Main Workspace Tabs
# ==============================================================================

tab_input, tab_plots, tab_analysis, tab_dist = st.tabs([
    "📝 1. Inspect, Edit & Add Raw Data",
    "📊 2. Interactive PSD & Rosin-Rammler Plots",
    "📋 3. Full Analysis Table (d', n, d10-d95, Shape)",
    "⚖️ 4. Percentage & Cumulative Distributions"
])

with tab_input:
    st.subheader("Inspect, Edit, or Add Sieve Mass Data (g)")

    # --- On-the-Fly Sieve Column Adder / Remover ---
    with st.expander("➕ / ➖ Add or Remove Sieve Size Columns in Current Table", expanded=False):
        col_add1, col_add2, col_rem1, col_rem2 = st.columns([2, 1, 2, 1])
        with col_add1:
            new_sieve_val = st.number_input(
                "New Sieve Aperture (um):", min_value=1.0, max_value=50000.0, value=500.0, step=50.0
            )
        with col_add2:
            st.write("")
            st.write("")
            if st.button("➕ Add Sieve Column", use_container_width=True):
                st.session_state["working_wide_df"] = insert_sieve_column_into_wide_df(
                    st.session_state["working_wide_df"], new_sieve_val
                )
                st.rerun()

        current_sieve_cols = [
            c for c in st.session_state["working_wide_df"].columns
            if c not in ['Sample', 'Category', 'pan (0 um)']
        ]
        with col_rem1:
            cols_to_drop = st.multiselect("Select Sieve Columns to Remove:", current_sieve_cols)
        with col_rem2:
            st.write("")
            st.write("")
            if st.button("🗑️ Remove Selected", use_container_width=True) and cols_to_drop:
                st.session_state["working_wide_df"] = st.session_state["working_wide_df"].drop(columns=cols_to_drop)
                st.rerun()

    st.markdown(
        "**Hybrid Data Editor:** Edit any weighed mass cell, add new sample rows using the **`+` row** at the bottom, "
        "or use the expander above to insert additional sieve apertures on the fly."
    )

    edited_wide_df = st.data_editor(
        st.session_state["working_wide_df"],
        num_rows="dynamic",
        use_container_width=True,
        key=f"editor_{st.session_state['source_sig']}"
    )

    # Convert edited table back to long format and run calculations
    df_active_long = wide_to_long_df(edited_wide_df)

    if df_active_long.empty:
        st.warning("No valid sample rows found. Please enter sample names and weighed masses above.")
        st.stop()

    _, processed_samples = calculate_distributions(df_active_long)
    if not processed_samples:
        st.warning("Total mass across sieves is 0 g for all rows. Enter positive sieve masses (g) in the table above to run the analysis.")
        st.stop()

    df_analysis = run_psd_analysis_dataset(processed_samples)
    st.success(
        f"✅ Live pipeline active: **{len(df_analysis)} valid samples** across "
        f"**{len(edited_wide_df.columns) - 2} sieve stages (including pan)**."
    )

# --- Sidebar Filters ---
st.sidebar.divider()
st.sidebar.header("3. Filter Active Samples")
categories = list(df_analysis['Category'].unique())
selected_cats = st.sidebar.multiselect("Select Categories", categories, default=categories[:1])

filtered_analysis = df_analysis[df_analysis['Category'].isin(selected_cats)]
available_samples = list(filtered_analysis['Sample'].unique())
selected_samples = st.sidebar.multiselect(
    "Select Samples to Plot",
    available_samples,
    default=available_samples[:min(5, len(available_samples))]
)

with tab_plots:
    col_plot, col_opts = st.columns([3, 1])
    with col_opts:
        st.subheader("Plot Settings")
        show_rr_overlay = st.checkbox("Overlay Rosin-Rammler Fit Curve", value=False)
        show_d50_line = st.checkbox("Show d50 Reference Line (50%)", value=True)

    with col_plot:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(9, 5.5))

        max_plot_sieve = 6000.0
        for item in processed_samples:
            if item['Category'] in selected_cats and item['Sample'] in selected_samples:
                max_plot_sieve = max(max_plot_sieve, float(item['Sieves_um'].max()))
                label = f"{item['Sample']} ({item['Category']})"
                plot_psd_curve(ax, item['Sieves_um'], item['Cum_Passing'], label=label)

                if show_rr_overlay:
                    row_match = df_analysis[
                        (df_analysis['Sample'] == item['Sample']) &
                        (df_analysis['Category'] == item['Category'])
                    ].iloc[0]
                    d_smooth = np.geomspace(10.0, float(item['Sieves_um'].max()), 300)
                    q_rr = rosin_rammler_undersize(d_smooth, row_match["d' (um)"], row_match['n'])
                    ax.plot(d_smooth, q_rr, linestyle=':', alpha=0.7, label=f"{item['Sample']} RR Fit")

        if show_d50_line:
            ax.axhline(50.0, color='gray', linestyle='--', linewidth=1, label='$d_{50}$ (50% Passing)')

        finalize_axis(ax, f"Particle Size Distributions ({sheet_choice})")
        ax.set_xlim(10.0, max_plot_sieve * 1.05)
        st.pyplot(fig)

with tab_analysis:
    st.subheader("Complete Analysis Parameters")
    st.dataframe(filtered_analysis.style.format(precision=2), use_container_width=True)

    csv_bytes = filtered_analysis.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Analysis Table as CSV",
        data=csv_bytes,
        file_name=f"psd_analysis_{sheet_choice.lower()}.csv",
        mime="text/csv"
    )

with tab_dist:
    st.subheader("Calculated Percentage Mass & Cumulative Passing")
    dist_rows = []
    for item in processed_samples:
        if item['Category'] in selected_cats:
            for s_um, cum_q in zip(item['Sieves_um'], item['Cum_Passing']):
                dist_rows.append({
                    'Sample': item['Sample'],
                    'Category': item['Category'],
                    'Total_Mass_g': item['Total_Mass_g'],
                    'Sieve_um': s_um,
                    'Cum_Passing_%': cum_q
                })
    st.dataframe(pd.DataFrame(dist_rows), use_container_width=True)