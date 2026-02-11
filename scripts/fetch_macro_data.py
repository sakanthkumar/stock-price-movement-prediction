import os
import requests
import pandas as pd
from dotenv import load_dotenv
import logging

# Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FRED_API_KEY = os.getenv("FRED_KEY")

def fetch_macro_data():
    """
    Fetches macroeconomic indicators from FRED API.
    Returns a dictionary of values.
    """
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations"
        
        indicators = {
            "GDP": "GDP",              # Gross Domestic Product
            "CPI": "CPIAUCSL",         # Consumer Price Index
            "UNEMPLOYMENT": "UNRATE"   # Unemployment Rate
        }

        macro_data = {}

        for key, series_id in indicators.items():
            params = {
                "series_id": series_id,
                "api_key": FRED_KEY,
                "file_type": "json",
                "limit": 1,
                "sort_order": "desc"
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                latest_value = data["observations"][0]["value"]
                macro_data[key] = float(latest_value) if latest_value != "." else None
            else:
                logging.warning(f"Failed to fetch {key}, status: {response.status_code}")
                macro_data[key] = None

        logging.info("Fetched macroeconomic indicators successfully")
        return macro_data

    except Exception as e:
        logging.error(f"Error fetching macro data: {e}")
        return {}

# ====== Main Execution ======
if __name__ == "__main__":
    # Example stock fetching (existing part of your script)
    logging.info("Fetching stock data for AAPL...")
    df = yf.download("AAPL", period="1mo", interval="1d")
    df.reset_index(inplace=True)

    # Call macro fetch
    macro = fetch_macro_data()

    # Add macro values to every row in dataset
    for key, val in macro.items():
        df[key] = val

    logging.info(f"Final dataset with macro data:\n{df.head()}")
    df.to_csv("combined_dataset.csv", index=False)