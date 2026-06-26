import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# We will just mimic the logic up to the residual training to see the values
sys.path.append(os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast"))
from quartz_solar_forecast.forecasts.v2 import TryolabsSolarPowerPredictor

df = pd.DataFrame({'a': [1,2,3]})
print("Hello world")
