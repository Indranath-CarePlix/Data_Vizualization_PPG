import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.header("CarePlix Sequential PPG Signal Visualisation", divider='rainbow')

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


# Function to plot sequential PPG signal
def plot_sequential_ppg_signal(df, current_row):
    if 0 <= current_row < len(df):
        
        row_index = df.iloc[current_row]['Row_Index']
        device = df.iloc[current_row]['Device_Name']
        ppg_signal = df.iloc[current_row]['PPG']
        sbp = df.iloc[current_row]['SBP']
        dbp = df.iloc[current_row]['DBP']

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(ppg_signal))), y=ppg_signal[::-1], mode='lines', 
                                 name='PPG Signal', line=dict(color='lightyellow')))
        label_text = f"Row_Index: {row_index}, <br> Device_Name: {device}, SBP: {sbp}, <br> DBP: {dbp}"
        fig.add_annotation(x=1.0, y=1.2, xref='paper', yref='paper', text=label_text, showarrow=False, font=dict(size=10))
        fig.update_layout(title="Sequential PPG Signal with Row_Index, Device",
                          xaxis_title="PPG Array Index",
                          yaxis_title="PPG Value",
                          template="plotly_dark",autosize = True, width=2000)
        st.plotly_chart(fig)
    else:
        st.write("End of dataframe reached.")


# Streamlit app
def main():
    feather_dir_path_full = "./Dataset/"
    obs_file_path = "./observations.txt"

    # Filter out only the feather files and extract the base file name
    feather_files = [os.path.splitext(file)[0] for file in os.listdir(feather_dir_path_full) if file.endswith('.feather')]

    # Add a default option
    feather_files.insert(0, "Select a File")
    
    # Dropdown/select box for choosing dataframe index
    selected_file = st.selectbox("Select the Dataframe:", feather_files)

    # Check if a value has been selected
    if selected_file != "Select a File":

        # Progress bar while loading dataframe
        progress_bar = st.progress(0)

         # Load dataframe (this will only be executed once for each unique index)
        df_processed = load_dataframe_full(selected_file, feather_dir_path_full)

        # Update progress bar to completion
        progress_bar.progress(100)

        "...Dataframe Loaded"

        st.text("CarePlix Sequential PPG Signal Visualization")

        # User input for selecting the starting row index
        start_row = st.number_input("Select Row Index:", 0, len(df_processed) - 1, 0)


        # Show the graph immediately when the user enters the row index
        plot_sequential_ppg_signal(df_processed, start_row)




# Run the Streamlit app
if __name__ == "__main__":
    main()
