import requests
import xml.etree.ElementTree as ET

url = "https://oedi-data-lake.s3.amazonaws.com/?prefix=pvdaq/2023-solar-data-prize/9069_OEDI/"
response = requests.get(url)

if response.status_code == 200:
    root = ET.fromstring(response.content)
    namespace = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    power_files = []
    for contents in root.findall('s3:Contents', namespace):
        key = contents.find('s3:Key', namespace).text
        if 'power' in key.lower() or 'meter' in key.lower() or 'ac' in key.lower():
            power_files.append(key)
    
    print(f"Found {len(power_files)} potential power files:")
    for key in power_files[:20]: 
        print(key)
else:
    print(f"Failed to fetch: {response.status_code}")
