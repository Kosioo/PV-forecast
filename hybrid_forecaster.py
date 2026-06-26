import pandas as pd
import numpy as np
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import xgboost as xgb

class PVLibPhysicalModel:
    def __init__(self, lat, lon, kwp, tilt, azimuth):
        self.lat = lat
        self.lon = lon
        self.kwp = kwp
        self.tilt = tilt
        self.azimuth = azimuth
        
        self.location = Location(latitude=lat, longitude=lon, tz='UTC')
        
        # Generic PVWatts system setup
        self.system = PVSystem(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            module_parameters={'pdc0': kwp * 1000, 'gamma_pdc': -0.004}, # W
            inverter_parameters={'pdc0': kwp * 1000, 'eta_inv_nom': 0.96}, # W
            temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        )
        self.mc = ModelChain(self.system, self.location, aoi_model='physical', spectral_model='no_loss')

    def predict(self, weather_df):
        """
        weather_df must be indexed by a timezone-aware DatetimeIndex
        Required columns: ghi, dni, dhi, temp_air, wind_speed
        """
        self.mc.run_model(weather_df)
        ac_power_w = self.mc.results.ac
        
        if isinstance(ac_power_w, pd.Series):
            ac_power_w = ac_power_w.fillna(0)
            
        ac_power_kw = ac_power_w / 1000.0
        
        # Clip to plant capacity (rough physical limit)
        ac_power_kw = np.clip(ac_power_kw, 0, self.kwp)
        return ac_power_kw

class MLResidualModel:
    def __init__(self):
        # We use XGBoost as it is robust and fast. LightGBM is similar but XGBoost handles NaNs well out of the box.
        self.model = xgb.XGBRegressor(
            n_estimators=150, 
            max_depth=5, 
            learning_rate=0.08, 
            random_state=42,
            subsample=0.8,
            colsample_bytree=0.8
        )
        self.feature_cols = [
            'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
            'surface_pressure', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
            'cloud_cover_high', 'wind_speed_10m', 'wind_direction_10m', 'is_day',
            'direct_radiation', 'diffuse_radiation', 'shortwave_radiation',
            'direct_normal_irradiance', 'terrestrial_radiation',
            'date_month', 'date_day', 'date_hour', 'date_minute'
        ]

    def _prepare_features(self, df):
        X = df[self.feature_cols].copy()
        return X

    def fit(self, df, residuals):
        X = self._prepare_features(df)
        self.model.fit(X, residuals)

    def predict(self, df):
        X = self._prepare_features(df)
        return self.model.predict(X)

class HybridForecaster:
    def __init__(self, lat, lon, kwp, tilt, azimuth):
        self.physical_model = PVLibPhysicalModel(lat, lon, kwp, tilt, azimuth)
        self.ml_model = MLResidualModel()
        
    def fit(self, df_weather, actual_power_kw):
        """
        df_weather must have the correct features and a localized DatetimeIndex for pvlib.
        actual_power_kw is a pandas Series aligned with df_weather.
        """
        # 1. Get physical prediction
        phys_pred = self.physical_model.predict(df_weather)
        
        # 2. Calculate residuals
        residuals = actual_power_kw - phys_pred
        
        # 3. Train ML model on residuals
        self.ml_model.fit(df_weather, residuals)
        
    def predict(self, df_weather):
        phys_pred = self.physical_model.predict(df_weather)
        ml_res = self.ml_model.predict(df_weather)
        
        final_pred = phys_pred + ml_res
        
        # Apply strict logical bounds
        final_pred = np.maximum(0, final_pred) # No negative generation
        final_pred = np.minimum(final_pred, self.physical_model.kwp) # Cap at plant capacity
        final_pred = np.where(df_weather['is_day'] == 0, 0, final_pred) # Night is zero
        
        return phys_pred, ml_res, final_pred
