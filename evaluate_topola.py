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
        return {'mae': 0, 'mae_pct_cap': 0, 'mae_pct_act': 0, 'rmse': 0, 'r2': 0}
    
    errors = df_day[target_col] - df_day[pred_col]
    abs_errors = np.abs(errors)
    
    mae = abs_errors.mean()
    mae_pct_cap = (mae / capacity) * 100.0
    actual_mean = df_day[target_col].mean()
    mae_pct_act = (mae / actual_mean) * 100.0 if actual_mean > 0 else 0
    
    rmse = np.sqrt((errors ** 2).mean())
    
    ss_res = (errors ** 2).sum()
    ss_tot = ((df_day[target_col] - df_day[target_col].mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {'mae': mae, 'mae_pct_cap': mae_pct_cap, 'mae_pct_act': mae_pct_act, 'rmse': rmse, 'r2': r2}

def parse_topola_data():
    """Parse Topola1.xlsx and return 15-min power data with tz-NAIVE UTC index.
    
    The Excel file has 2 columns (after skipping row 0 header):
      - Column 0: datetime in local Sofia time (parsed as datetime64 by pandas)
      - Column 1: power output in MW (float64)
    
    Returns a DataFrame with tz-naive UTC DatetimeIndex and 'actual_power_kw' column.
    This ensures clean joins with weather data (also tz-naive UTC from Open-Meteo).
    """
    print("Loading Topola1.xlsx...")
    df_raw = pd.read_excel(EXCEL_FILE, skiprows=1, header=None)
    df_raw.columns = ['date_local', 'capacity_mw']
    
    # The datetime column is already parsed as datetime64 by pandas from Excel
    if not pd.api.types.is_datetime64_any_dtype(df_raw['date_local']):
        df_raw['date_local'] = pd.to_datetime(
            df_raw['date_local'].astype(str).str.strip(), 
            errors='coerce', dayfirst=True
        )
    df_raw = df_raw.dropna(subset=['date_local'])
    
    # Power is already float64 in MW — convert to kW
    df_raw['actual_power_kw'] = pd.to_numeric(df_raw['capacity_mw'], errors='coerce') * 1000.0
    df_raw = df_raw.dropna(subset=['actual_power_kw'])
    df_raw.loc[df_raw['actual_power_kw'] < 0, 'actual_power_kw'] = 0
    
    # Convert local Sofia time to UTC (tz-naive result for clean joins)
    # ambiguous='NaT' drops the ~4 intervals during DST fall-back transition (Oct 26 3:00 AM)
    df_raw['date_local'] = df_raw['date_local'].dt.tz_localize(
        TIMEZONE, ambiguous='NaT', nonexistent='shift_forward'
    )
    df_raw = df_raw.dropna(subset=['date_local'])
    df_raw['date_utc'] = df_raw['date_local'].dt.tz_convert('UTC').dt.tz_localize(None)
    
    df_power = df_raw[['date_utc', 'actual_power_kw']].copy()
    df_power.set_index('date_utc', inplace=True)
    
    # Resample to 15-min — use mean for any duplicate timestamps
    df_15 = df_power.resample('15min').mean()
    # Do NOT interpolate — we want real data only, not fabricated values
    df_15 = df_15.dropna()
    
    # Drop any future rows
    now_utc = pd.Timestamp.now(tz='UTC').tz_localize(None)
    df_15 = df_15[df_15.index <= now_utc]
    
    # Drop days where the plant was genuinely offline (max power < 50 kW)
    # User confirmed only Jan 22-23, 2026 were offline
    daily_max = df_15['actual_power_kw'].resample('D').max()
    valid_days = daily_max[daily_max > 50].index.date
    df_15 = df_15[np.isin(df_15.index.date, valid_days)]
    
    print(f"  Loaded {len(df_15)} valid 15-min intervals across {len(np.unique(df_15.index.date))} days")
    print(f"  Date range: {df_15.index.min()} to {df_15.index.max()}")
    print(f"  Power range: {df_15['actual_power_kw'].min():.1f} — {df_15['actual_power_kw'].max():.1f} kW")
    
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
    
    # Both df_weather and df_15 are tz-naive UTC — clean inner join
    dataset = df_weather.join(df_15, how='inner')
    dataset = dataset.dropna(subset=['actual_power_kw'])
    
    # Localize for pvlib (requires tz-aware index)
    dataset.index = dataset.index.tz_localize('UTC')
    
    print(f"  Merged dataset: {len(dataset)} intervals")
    
    # Train / Test Split — use chronological split (last 25% for test)
    n = len(dataset)
    split_idx = int(n * 0.75)
    df_train = dataset.iloc[:split_idx].copy()
    df_test = dataset.iloc[split_idx:].copy()
    
    print(f"Training on {len(df_train)} intervals, Testing on {len(df_test)} intervals...")
    print(f"  Train period: {df_train.index.min().date()} to {df_train.index.max().date()}")
    print(f"  Test period:  {df_test.index.min().date()} to {df_test.index.max().date()}")
    
    forecaster = HybridForecaster(LATITUDE, LONGITUDE, CAPACITY_KWP, TILT, ORIENTATION)
    forecaster.fit(df_train, df_train['actual_power_kw'])
    
    phys_pred, ml_res, final_pred = forecaster.predict(df_test)
    df_test['pred_phys'] = phys_pred
    df_test['pred_hybrid'] = final_pred
    
    # === 15-Minute Results ===
    m_phys = calculate_metrics(df_test, 'pred_phys')
    m_hyb = calculate_metrics(df_test, 'pred_hybrid')
    
    print("\n" + "=" * 70)
    print(" 15-MINUTE RESULTS (Daylight Only)")
    print("=" * 70)
    print(f"{'Metric':<25} {'Physical Model':>18} {'Hybrid Model':>18}")
    print("-" * 70)
    print(f"{'MAE (kW)':<25} {m_phys['mae']:>18.2f} {m_hyb['mae']:>18.2f}")
    print(f"{'MAE (% of Capacity)':<25} {m_phys['mae_pct_cap']:>17.2f}% {m_hyb['mae_pct_cap']:>17.2f}%")
    print(f"{'MAE (% of Actual)':<25} {m_phys['mae_pct_act']:>17.2f}% {m_hyb['mae_pct_act']:>17.2f}%")
    print(f"{'RMSE (kW)':<25} {m_phys['rmse']:>18.2f} {m_hyb['rmse']:>18.2f}")
    print(f"{'R²':<25} {m_phys['r2']:>18.4f} {m_hyb['r2']:>18.4f}")
    
    # === 1-Hour Aggregation ===
    df_test_1h = df_test[['actual_power_kw', 'pred_phys', 'pred_hybrid', 'is_day']].resample('1h').mean()
    df_test_1h['is_day'] = df_test_1h['is_day'].round()
    
    m_phys_1h = calculate_metrics(df_test_1h, 'pred_phys')
    m_hyb_1h = calculate_metrics(df_test_1h, 'pred_hybrid')
    
    print("\n" + "=" * 70)
    print(" 1-HOUR RESULTS (Daylight Only)")
    print("=" * 70)
    print(f"{'Metric':<25} {'Physical Model':>18} {'Hybrid Model':>18}")
    print("-" * 70)
    print(f"{'MAE (kW)':<25} {m_phys_1h['mae']:>18.2f} {m_hyb_1h['mae']:>18.2f}")
    print(f"{'MAE (% of Capacity)':<25} {m_phys_1h['mae_pct_cap']:>17.2f}% {m_hyb_1h['mae_pct_cap']:>17.2f}%")
    print(f"{'MAE (% of Actual)':<25} {m_phys_1h['mae_pct_act']:>17.2f}% {m_hyb_1h['mae_pct_act']:>17.2f}%")
    print(f"{'RMSE (kW)':<25} {m_phys_1h['rmse']:>18.2f} {m_hyb_1h['rmse']:>18.2f}")
    print(f"{'R²':<25} {m_phys_1h['r2']:>18.4f} {m_hyb_1h['r2']:>18.4f}")

    # Save model for later use (schedule generation)
    forecaster.ml_model.model.save_model("topola_hybrid_model.json")
    print("\nSaved trained residual model to 'topola_hybrid_model.json'")

if __name__ == "__main__":
    main()