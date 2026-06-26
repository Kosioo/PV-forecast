import pandas as pd

data = {'time_str': ['1.6.2025 10:00', '1.6.2025 10:15'], 'capacity_mw': ['3,6612', '3,7000']}
df_raw = pd.DataFrame(data)

df_raw['date_local'] = pd.to_datetime(df_raw['time_str'], format='%d.%m.%Y %H:%M', errors='coerce')
if df_raw['capacity_mw'].dtype == object:
    df_raw['capacity_mw'] = df_raw['capacity_mw'].str.replace(',', '.')
df_raw['actual_power_kw'] = pd.to_numeric(df_raw['capacity_mw'], errors='coerce') * 1000.0

print(df_raw)
