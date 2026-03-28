# test_oanda.py
import requests

API_KEY = "fceae94e861642af6c9d9de1bf6a319d-52074b19e787811450bc44152cb71e78"
BASE_URL = "https://api-fxpractice.oanda.com/v3"

# Test 1: Check if API key works by getting accounts
headers = {"Authorization": f"Bearer {API_KEY}"}
response = requests.get(f"{BASE_URL}/accounts", headers=headers)

print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    print("✅ API Key is valid!")
    print("Accounts:", response.json())
else:
    print("❌ API Key invalid or expired")
    print("Response:", response.text)

# Test 2: Try a simple candle request with a known good pair
if response.status_code == 200:
    instr_response = requests.get(
        f"{BASE_URL}/instruments/EUR_USD/candles",
        headers=headers,
        params={"granularity": "H1", "count": 10}
    )
    print(f"\nCandle Request Status: {instr_response.status_code}")
    if instr_response.status_code == 200:
        print("✅ Can fetch candles!")
    else:
        print("❌ Candle fetch failed:", instr_response.text)