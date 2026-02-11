import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data/processed/AAPL_with_macro_sentiment.csv', parse_dates=['date'])
df['macro_value'] = df['macro_value'].fillna(method='ffill').fillna(method='bfill')

# For sentiment, if missing, fill with neutral 0 or interpolate if you prefer
df['sentiment'] = df['sentiment'].fillna(0)

# Save cleaned data for later steps
df.to_csv('data/processed/AAPL_with_macro_sentiment_clean.csv', index=False)

print("Missing values handled and cleaned data saved.")

# Load the processed dataset
data_path = "data/processed/AAPL_with_macro_sentiment.csv"
df = pd.read_csv(data_path, parse_dates=['date'])
plt.figure(figsize=(12,6))
plt.plot(df['date'], df['Close'], label='Close Price')
plt.title('AAPL Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Plot macroeconomic values (with missing data)
plt.figure(figsize=(12,6))
plt.plot(df['date'], df['macro_value'], label='Macro Value (UNRATE)')
plt.title('Macroeconomic Indicator Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plot sentiment if available
if df['sentiment'].notna().any():
    plt.figure(figsize=(12,6))
    plt.plot(df['date'], df['sentiment'], label='Sentiment')
    plt.title('Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.legend()
    plt.show()
else:
    print("No sentiment data available to plot.")
print("Data loaded successfully!")
print(df.head())
print(df.info())
