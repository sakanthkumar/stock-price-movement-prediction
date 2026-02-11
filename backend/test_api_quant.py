import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_analyze_stock():
    ticker = "AAPL"
    print(f"Testing /analyze_stock/{ticker}...")
    try:
        response = requests.get(f"{BASE_URL}/analyze_stock/{ticker}")
        if response.status_code == 200:
            data = response.json()
            print("Response:")
            print(json.dumps(data, indent=2))
            
            # Verify structure
            signals = data.get("signals", {})
            swing = signals.get("swing", {})
            intraday = signals.get("intraday", {})
            
            if "signal" in swing and "action" in swing and "confidence" in swing:
                print("✅ Swing structure is valid.")
            else:
                print("❌ Swing structure invalid.")

            if "signal" in intraday and intraday["signal"] != "No Model":
                print(f"✅ Intraday signal found: {intraday['signal']}")
            else:
                print(f"❌ Intraday signal issue: {intraday.get('signal')}")

        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_analyze_stock()
