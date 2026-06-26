import numpy as np
import xgboost as xgb

X = np.random.rand(100, 10)
y = np.full(100, -2700.0)

model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
model.fit(X, y)

pred = model.predict(X[0:1])
print(f"Target: -2700, Predicted: {pred[0]}")
