import pandas as pd
import numpy as np
import warnings
from hybrid_forecaster import HybridForecaster
from evaluate_all_plants import get_archived_weather_15min
warnings.filterwarnings("ignore")

# ================= Configuration =================
EXCEL_FILE = "Topola1.xlsx"
LATITUDE = 41.509238
LONGITUDE = 23.796700
CAPACITY_KWP = 5000.0
TILT = 38.0  
ORIENTATION = 180.0
TIMEZONE = 'Europe/Sofia'
# =================================================

def calculate_metrics(df_test, pred_col, target_col='actual_power_kw', capacity=CAPACITY_KWP):
    mask = df_test['is_day'] == 1
    df_day = df_test[mask]
    if len(df_day) == 0:
        return 0, 0, 0
    mae = np.mean(np.abs(df_day[target_col] - df_day[pred_col]))
    mae_perc_cap = (mae / capacity) * 100.0
    actual_mean = df_day[target_col].mean()
    mae_perc_actual = (mae / actual_mean) * 100.0 if actual_mean > 0 else 0
    return mae, mae_perc_cap, mae_perc_actual

def parse_topola_data():
    print("Loading Topola1.xlsx...")
    try:
        df_raw = pd.read_excel(EXCEL_FILE, skiprows=1)
    except:
        df_raw = pd.read_csv(EXCEL_FILE, skiprows=1)
        
    df_raw.columns = ['time_str', 'capacity_mw']
    df_raw['time_str'] = df_raw['time_str'].astype(str).str.strip()
    df_raw['date_local'] = pd.to_datetime(df_raw['time_str'], errors='coerce', dayfirst=True)
    df_raw = df_raw.dropna(subset=['date_local'])
    
    import re
    def parse_mw(x):
        if isinstance(x, str):
            x = x.replace(',', '.')
            x = re.sub(r'[^\d\.]', '', x)
        try:
            return float(x) * 1000.0 # MW to kW
        except:
            return np.nan
            
    df_raw['actual_power_kw'] = df_raw['capacity_mw'].apply(parse_mw)
    df_raw = df_raw.dropna(subset=['actual_power_kw'])
    
    df_raw['date_local'] = df_raw['date_local'].dt.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
    df_raw = df_raw.dropna(subset=['date_local'])
    df_raw['date_utc'] = df_raw['date_local'].dt.tz_convert('UTC')
    
    df_power = df_raw[['date_utc', 'actual_power_kw']].copy()
    df_power.set_index('date_utc', inplace=True)
    
    df_15 = df_power.resample('15min').mean()
    df_15 = df_15.interpolate()
    df_15.loc[df_15['actual_power_kw'] < 0, 'actual_power_kw'] = 0
    df_15 = df_15.dropna()
    
    # Drop any future rows that might be in the Excel file (e.g. empty month templates)
    now_utc = pd.Timestamp.now(tz='UTC')
    df_15 = df_15[df_15.index <= now_utc]
    
    return df_15

def main():
    print("=====================================================")
    print(f" Hybrid Pipeline Evaluation: Topola (5MW)")
    print("=====================================================")
    
    df_15 = parse_topola_data()
    
    start_date = df_15.index.min().strftime('%Y-%m-%d')
    end_date = df_15.index.max().strftime('%Y-%m-%d')
    print(f"Fetching historical weather from {start_date} to {end_date}...")
    df_weather = get_archived_weather_15min(LATITUDE, LONGITUDE, start_date, end_date)
    
    # Merge
    dataset = df_weather.copy()
    dataset.index = dataset.index.tz_localize('UTC')
    dataset = dataset.join(df_15, how='inner')
    dataset = dataset.dropna(subset=['actual_power_kw'])
    
    # Train / Test Split
    dataset['day_of_year'] = dataset.index.dayofyear
    train_mask = (dataset['day_of_year'] % 2 == 0)
    test_mask = ~train_mask
    
    df_train = dataset[train_mask].copy()
    df_test = dataset[test_mask].copy()
    
    print(f"Training on {len(df_train)} intervals, Testing on {len(df_test)} intervals...")
    forecaster = HybridForecaster(LATITUDE, LONGITUDE, CAPACITY_KWP, TILT, ORIENTATION)
    forecaster.fit(df_train, df_train['actual_power_kw'])
    
    phys_pred, ml_res, final_pred = forecaster.predict(df_test)
    df_test['pred_phys'] = phys_pred
    df_test['pred_hybrid'] = final_pred
    
    mae_phys, cap_phys, act_phys = calculate_metrics(df_test, 'pred_phys')
    mae_hyb, cap_hyb, act_hyb = calculate_metrics(df_test, 'pred_hybrid')
    print("\n--- 15-MINUTE RESULTS (Daylight Only) ---")
    print(f"[Physical Model] MAE: {mae_phys:.2f} kW | {cap_phys:.2f}% of Cap | {act_phys:.2f}% of Actual")
    print(f"[Hybrid Model]   MAE: {mae_hyb:.2f} kW | {cap_hyb:.2f}% of Cap | {act_hyb:.2f}% of Actual")
    
    # 1-Hour Aggregation
    df_test_1h = df_test[['actual_power_kw', 'pred_phys', 'pred_hybrid', 'is_day']].resample('1h').mean()
    df_test_1h['is_day'] = df_test_1h['is_day'].round()
    
    mae_phys_1h, cap_phys_1h, act_phys_1h = calculate_metrics(df_test_1h, 'pred_phys')
    mae_hyb_1h, cap_hyb_1h, act_hyb_1h = calculate_metrics(df_test_1h, 'pred_hybrid')
    print("\n--- 1-HOUR RESULTS (Daylight Only) ---")
    print(f"[Physical Model] MAE: {mae_phys_1h:.2f} kW | {cap_phys_1h:.2f}% of Cap | {act_phys_1h:.2f}% of Actual")
    print(f"[Hybrid Model]   MAE: {mae_hyb_1h:.2f} kW | {cap_hyb_1h:.2f}% of Cap | {act_hyb_1h:.2f}% of Actual")

    # Save model for later use (schedule generation)
    forecaster.ml_model.model.save_model("topola_hybrid_model.json")
    print("\nSaved trained residual model to 'topola_hybrid_model.json'")

if __name__ == "__main__":
    main()