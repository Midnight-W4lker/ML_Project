import yfinance as yf
import pandas as pd

ticker = "AAPL"
start_date = "2023-01-01"
end_date = "2023-01-10"

print(f"Downloading data for {ticker}...")
data = yf.download(ticker, start=start_date, end=end_date)
print("Columns:", data.columns)
print("Head:\n", data.head())
print("Dtypes:\n", data.dtypes)

# Simulate saving and loading
data.reset_index(inplace=True)
data.to_csv("test_data.csv", index=False)

df = pd.read_csv("test_data.csv")
print("\nLoaded Columns:", df.columns)
print("Loaded Head:\n", df.head())
print("Loaded Dtypes:\n", df.dtypes)
