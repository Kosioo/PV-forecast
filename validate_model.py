import pandas as pd
import numpy as np

def calculate_errors(actual: pd.Series, forecast: pd.Series) -> dict:
    """Calculate MAE, RMSE, and MAPE."""
    # Filter out night time where actual and forecast are both very small
    threshold = actual.max() * 0.05
    mask = (actual > threshold) | (forecast > threshold)
    a = actual[mask]
    f = forecast[mask]
    
    if len(a) == 0:
        return {"MAE": 0, "RMSE": 0, "MAPE": 0}
        
    mae = np.mean(np.abs(a - f))
    rmse = np.sqrt(np.mean((a - f)**2))
    mape = np.mean(np.abs((a - f) / a)) * 100
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def main():
    print("--- Historical Day-Ahead Model Validation ---")
    print("Loading actual historical PV production (using simulated data for Simon Solar Farm)...")
    
    # In a real scenario, you would load your OEDI or Bulgarian CSV here:
    # df_actual = pd.read_csv("9069_Site_Power_Nov2023.csv")
    
    # For demonstration, let's load our 15-minute forecast output 
    # and simulate an "actual" generation by adding noise and cloud events
    try:
        df = pd.read_csv("forecast_15min_output.csv")
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        print("Please run forecast_15min.py first to generate the base data.")
        return
        
    # Simulate Actual Production (Adding 10% random noise + some cloud dropouts)
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, size=len(df))
    df['actual_power_kw'] = df['power_kw'] * (1 + noise)
    
    # Simulate a sudden cloud event at peak noon
    peak_idx = df['power_kw'].idxmax()
    if pd.notna(peak_idx):
        df.loc[peak_idx-2:peak_idx+2, 'actual_power_kw'] *= 0.5
        
    df.loc[df['actual_power_kw'] < 0, 'actual_power_kw'] = 0
    
    print("Calculating error metrics between Forecast and Actuals...")
    errors = calculate_errors(df['actual_power_kw'], df['power_kw'])
    
    print("\nValidation Results:")
    print(f"Mean Absolute Error (MAE): {errors['MAE']:.2f} kW")
    print(f"Root Mean Square Error (RMSE): {errors['RMSE']:.2f} kW")
    print(f"Mean Absolute Percentage Error (MAPE): {errors['MAPE']:.2f}%")
    
    print("\nTo validate your own data:")
    print("1. Replace the 'actual_power_kw' column with your true historical measurements.")
    print("2. Run the forecast for the same historical dates.")
    print("3. Execute this script to calculate the Day-Ahead Market error.")

if __name__ == "__main__":
    main()
