import pandas as pd
import os

files = {
    'Netherlands': 'PV_measurements/pvoutput_data_netherlands1.csv',
    'NC-1': 'PV_measurements/14200_all_channels_20220101_20221231.csv',
    'NC-2': 'PV_measurements/14201_all_channels_20220101_20221231.csv',
    'Germany': 'PV_measurements/pvoutput_data_germany1.csv'
}

for name, path in files.items():
    print(f"\n--- {name} ---")
    if not os.path.exists(path):
        print("File not found.")
        continue
    
    df = pd.read_csv(path, nrows=5)
    print("Columns:", df.columns.tolist())
    
    # Read full to get dates
    df = pd.read_csv(path)
    if 'Date' in df.columns and 'Time' in df.columns:
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    elif 'utc_measured_on' in df.columns:
        df['datetime'] = pd.to_datetime(df['utc_measured_on'])
        
    print("Start Date:", df['datetime'].min())
    print("End Date:", df['datetime'].max())
    print("Row Count:", len(df))
