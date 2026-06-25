import requests

url = "https://archive-api.open-meteo.com/v1/archive?latitude=35.8123&longitude=-78.7493&start_date=2022-07-01&end_date=2022-07-02&minutely_15=temperature_2m,relative_humidity_2m"
print("Testing archive API minutely_15...")
response = requests.get(url)
print(response.status_code)
if response.status_code != 200:
    print(response.json())
else:
    print("Keys:", response.json().keys())
