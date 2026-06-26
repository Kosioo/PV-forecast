import sys
import os
import pandas as pd
import xgboost as xgb
sys.path.append(os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast"))
from quartz_solar_forecast.forecasts.v2 import TryolabsSolarPowerPredictor

predictor = TryolabsSolarPowerPredictor()
predictor.load_model()

# Create dummy weather row for perfect sunny noon
data = {
    'latitude_rounded': [41.5],
    'longitude_rounded': [23.8],
    'orientation': [180.0],
    'tilt': [38.0],
    'kwp': [5000.0], # Topola
    'temperature_2m': [25.0],
    'relative_humidity_2m': [50.0],
    'dew_point_2m': [10.0],
    'precipitation': [0.0],
    'surface_pressure': [1000.0],
    'cloud_cover': [0.0],
    'cloud_cover_low': [0.0],
    'cloud_cover_mid': [0.0],
    'cloud_cover_high': [0.0],
    'wind_speed_10m': [5.0],
    'wind_direction_10m': [180.0],
    'is_day': [1],
    'direct_radiation': [1000.0], # STC Irradiance
    'diffuse_radiation': [0.0],
    'date_month': [6],
    'date_day': [26],
    'date_hour': [12]
}
df = pd.DataFrame(data)

base_model_path = os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast", "quartz_solar_forecast", "models", "model_10_202405.ubj")
base_model = xgb.XGBRegressor()
base_model.load_model(base_model_path)

pred = base_model.predict(df)[0]
print(f"Topola (5000 kwp) Base Model Prediction: {pred} (expected ~5.0 if MW, ~5000 if kW)")

df['kwp'] = 498.42
pred_nl = base_model.predict(df)[0]
print(f"Netherlands (498.42 kwp) Base Model Prediction: {pred_nl} (expected ~0.5 if MW, ~500 if kW)")

df['kwp'] = 14200.0
pred_nc = base_model.predict(df)[0]
print(f"NC (14200 kwp) Base Model Prediction: {pred_nc}")
