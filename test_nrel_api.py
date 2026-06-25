import requests

url = "https://developer.nrel.gov/api/solar/pvdaq/v3/site_data.json?api_key=DEMO_KEY&system_id=9069&start_date=2023-10-01&end_date=2023-10-02"
print(f"Fetching {url}")
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print("Success. Keys:", data.keys())
    if 'outputs' in data:
        print("Outputs keys:", data['outputs'].keys() if isinstance(data['outputs'], dict) else type(data['outputs']))
else:
    print(f"Failed: {response.status_code}")
    print(response.text[:200])
