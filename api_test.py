import requests

API_KEY = "DXI7AG2VVP5N5PCW"

url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=SPY&apikey={API_KEY}"

response = requests.get(url)
data = response.json()

print(data)