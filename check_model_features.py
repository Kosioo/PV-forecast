import xgboost as xgb

# Path to the downloaded model
model_path = r"c:\Users\konst\Documents\antigravity-mashup\PV-forecast\open-source-quartz-solar-forecast\quartz_solar_forecast\models\model_10_202405.ubj"

model = xgb.XGBRegressor()
model.load_model(model_path)

print("Features the model was trained on:")
print(model.feature_names_in_)
