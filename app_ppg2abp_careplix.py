import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
from scipy.signal import find_peaks
import neurokit2 as nk
import warnings
warnings.filterwarnings("ignore")


st.header("Sequential Signal Visualisation", divider='rainbow')

# For Random Signal
# Function to load dataframe based on the index
@st.cache_data
def load_dataframe_full(file_name, input_dir_feather):
    
    feather_file_name = f'{file_name}.feather'      ## changes required 
    feather_file_path = os.path.join(input_dir_feather, feather_file_name)

    df_feather = pd.read_feather(feather_file_path)

    # Reset the index to ensure Row_Index starts from 0
    df_feather.reset_index(drop=True, inplace=True)

    # Add a new column "Row_Id"
    df_feather['Row_Index'] = df_feather.reset_index().index
    return df_feather


# -------------------------------------------------------------------------------PPG----------------------------------------------------------------

# Function to plot sequential PPG signal
def plot_sequential_ppg_signal(df, current_row, signal_type):
    if 0 <= current_row < len(df):
        row = df.iloc[current_row]
        row_index = row['Row_Index']
        ppg_signal = row[signal_type]

        ## Using Scipy find_peaks
        peak_index_sc, _ = find_peaks(ppg_signal, prominence=0.1)

        ## Using neurokit
        _, systolic_info_nk = nk.ppg_peaks(ppg_signal, sampling_rate=125, correct_artifacts=False)

        ## Features:
        ## Get Max and Min PPG calculated
        max_peak = round(np.max(ppg_signal[peak_index_sc]),4)

        ## Median, Mean of peaks
        # Peaks
        if len(systolic_info_nk['PPG_Peaks']) == 0:
            median_peak = np.nan  # Assign NaN for missing values 
            mean_peak = np.nan
        else:
            # Calculate the median and mean
            median_peak = round(np.median(ppg_signal[systolic_info_nk['PPG_Peaks']]),3)
            mean_peak = round(np.mean(ppg_signal[systolic_info_nk['PPG_Peaks']]),3)


        # Exclude 'Row_Index' and 'PPG' columns from the label text
        label_columns = [col for col in df.columns if col not in ['Row_Index', 'PPG', 'ABP']]
        label_text = f"Row_Index: {row_index}, <br>Max_Peak: {max_peak}, Median_Peak: {median_peak}, Mean_Peak: {mean_peak},<br>"+ ", ".join([f"{col}: {row[col]}" for col in label_columns])

        # Trace for PPG Signal
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(ppg_signal))), y=ppg_signal, mode='lines', name='PPG_Signal', line=dict(color='ivory')))

        # Trace for Scipy Peaks from PPG
        fig.add_trace(go.Scatter(
        x=list(map(lambda x: np.arange(len(ppg_signal))[x], peak_index_sc)),
        y=list(map(lambda x: ppg_signal[x], peak_index_sc)),
        name='scipy_peaks',
        mode='markers',
        line_color='#f708e8'
        ))

        # Neurokit Peaks
        fig.add_trace(go.Scatter(
        x=list(map(lambda x: np.arange(len(ppg_signal))[x], systolic_info_nk['PPG_Peaks'])),
        y=list(map(lambda x: ppg_signal[x], systolic_info_nk['PPG_Peaks'])),
        name='nk_peaks',
        mode='markers',
        line_color='#00f900'
        ))


        # Add extra <br><br> for spacing
        fig.add_annotation(x=0.8, y=1.3, xref='paper', yref='paper', text=label_text, showarrow=False, font=dict(size=13), align='left')
        
        fig.update_layout(title="Sequential PPG Waveform: ",
                          xaxis_title="PPG Array Index",
                          yaxis_title="PPG Value",
                          template="plotly_dark",
                          autosize = True, width=1200,
                          margin=dict(l=50, r=0, t=120, b=0))
        st.plotly_chart(fig)
        st.text_area("PPG Values: ",list(ppg_signal), height=2000)

# -------------------------------------------------------------------------------ABP----------------------------------------------------------------

# Function to plot sequential ABP signal
def plot_sequential_abp_signal(df, current_row, signal_type):
    if 0 <= current_row < len(df):
        row = df.iloc[current_row]
        row_index = row['Row_Index']
        abp_signal = row[signal_type]

        # Peaks:
        ## Using Scipy find_peaks
        peak_index_sc, _ = find_peaks(abp_signal, prominence=0.1)

        ## Using neurokit
        _, systolic_info_nk = nk.ppg_peaks(abp_signal, sampling_rate=125, correct_artifacts=False)

        # Troughs

        ## Scipy:
        trough_index_sc, _ = find_peaks(-abp_signal, prominence=0.1)

        ## Using Neurokit
        abp_inv = -abp_signal
        abp_inv = abp_inv-min(abp_inv)
        # Experiment with different methods and adjust parameters as needed
        _, diastolic_info = nk.ppg_peaks(abp_inv, 125, correct_artifacts=False)

        ## Features:
        ## Get SBP and DBP calculated
        sbp_cal, dbp_cal = round(np.max(abp_signal[peak_index_sc]),0), round(np.min(abp_signal[trough_index_sc]),0)

        ## Median, Mean of peaks and trough
        # Peaks
        if len(systolic_info_nk['PPG_Peaks']) == 0:
            sbp_median = np.nan  # Assign NaN for missing values 
            sbp_mean = np.nan
        else:
            # Calculate the median and mean
            sbp_median = round(np.median(abp_signal[systolic_info_nk['PPG_Peaks']]),3)
            sbp_mean = round(np.mean(abp_signal[systolic_info_nk['PPG_Peaks']]),3)

        # Troughs
        if len(diastolic_info['PPG_Peaks']) == 0:
            dbp_median = np.nan  # Assign NaN for missing values 
            dbp_mean = np.nan
        else:
            # Calculate the median and mean
            dbp_median = round(np.median(abp_signal[diastolic_info['PPG_Peaks']]),3)
            dbp_mean = round(np.mean(abp_signal[diastolic_info['PPG_Peaks']]),3)       

        # Exclude 'Row_Index' and 'ABP' columns from the label text
        label_columns = [col for col in df.columns if col not in ['Row_Index', 'PPG', 'ABP']]
        label_text = f"Row_Index: {row_index}, <br>SBP_Cal: {sbp_cal}, SBP_median: {sbp_median}, SBP_mean: {sbp_mean}, <br>DBP_Cal: {dbp_cal}, DBP_median: {dbp_median}, DBP_mean: {dbp_mean}, <br>"+ ", ".join([f"{col}: {row[col]}" for col in label_columns])

        # Trace for ABP Signal
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(abp_signal))), y=abp_signal, mode='lines', name='ABP_Signal', line=dict(color='peru')))

        ## Peaks:
        # Trace for Scipy
        fig.add_trace(go.Scatter(
        x=list(map(lambda x: np.arange(len(abp_signal))[x], peak_index_sc)),
        y=list(map(lambda x: abp_signal[x], peak_index_sc)),
        name='scipy_peaks',
        mode='markers',
        line_color='#f708e8'
        ))

        # Neurokit Peaks
        fig.add_trace(go.Scatter(
        x=list(map(lambda x: np.arange(len(abp_signal))[x], systolic_info_nk['PPG_Peaks'])),
        y=list(map(lambda x: abp_signal[x], systolic_info_nk['PPG_Peaks'])),
        name='nk_peaks',
        mode='markers',
        line_color='#00f900'
        ))

        ## Troughs:
        # Trace for Scipy
        fig.add_trace(go.Scatter(
        x=list(map(lambda x: np.arange(len(abp_signal))[x], trough_index_sc)),
        y=list(map(lambda x: abp_signal[x], trough_index_sc)),
        name='scipy_troughs',
        mode='markers',
        line_color='#F9F500'
        ))

        # Neurokit
        fig.add_trace(go.Scatter(
        x=list(map(lambda x: np.arange(len(abp_signal))[x], diastolic_info['PPG_Peaks'])),
        y=list(map(lambda x: abp_signal[x], diastolic_info['PPG_Peaks'])),
        name='nk_troughs',
        mode='markers',
        line_color='#00E3F9'
        ))

        # Add extra <br><br> for spacing
        fig.add_annotation(x=0.8, y=1.3, xref='paper', yref='paper', text=label_text, showarrow=False, font=dict(size=13), align='left')
        
        fig.update_layout(title=f"Sequential ABP Waveform: ",
                          xaxis_title="ABP Array Index",
                          yaxis_title="ABP Value",
                          template="plotly_dark",
                          autosize = True, width=1200,
                          margin=dict(l=50, r=0, t=100, b=0))
        st.plotly_chart(fig)



# Streamlit app
def main():
    feather_dir_path_full = "./Dataset/"

    # Filter out only the feather files and extract the base file name
    feather_files = [os.path.splitext(file)[0] for file in os.listdir(feather_dir_path_full) if file.endswith('.feather')]

    # Add a default option
    feather_files.insert(0, "Select a File")
    
    # Dropdown/select box for choosing dataframe index
    selected_file = st.selectbox("Select the Dataframe:", feather_files)

    # Check if a value has been selected
    if selected_file != "Select a File":

        # Load dataframe only if the index has changed
        df_fea = load_dataframe_full(selected_file, feather_dir_path_full)

        # "Loading the Dataframe from .feather"

        # Progress bar while loading dataframe
        progress_bar = st.progress(0)

        # Load dataframe (this will only be executed once for each unique index)
        df_fea = load_dataframe_full(selected_file, feather_dir_path_full)

        # Update progress bar to completion
        progress_bar.progress(100)

        "...Dataframe Loaded"

        # User input for selecting the starting row index
        start_row = st.number_input("Select Row Index:", 0, len(df_fea) - 1, 0)

        # Check which columns are present
        if 'PPG' in df_fea.columns:
            st.text(f"Sequential PPG waveform Visualization")
            # Show the graph immediately when the user enters the row index
            plot_sequential_ppg_signal(df_fea, start_row, 'PPG')

            st.divider()  # 👈 Draws a horizontal rule
        if 'ABP' in df_fea.columns:
            st.text(f"Sequential ABP waveform Visualization")
            # Show the graph immediately when the user enters the row index
            plot_sequential_abp_signal(df_fea, start_row, 'ABP')
        
        ## Note: Add as per column based on the above

            st.divider()  # 👈 Draws a horizontal rule


# Run the Streamlit app
if __name__ == "__main__":
    main()

