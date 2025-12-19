import yfinance as yf
import pandas as pd
import os

def download_data(ticker, start_date, end_date, save_path='data/stock_data.csv'):
    """
    Downloads stock data from Yahoo Finance and saves it to a CSV file.
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        print("No data found for the given ticker and date range.")
        return None
    
    # Flatten MultiIndex columns if present (yfinance update)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Reset index to make Date a column
    data.reset_index(inplace=True)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")
    return data

def load_data(file_path='data/stock_data.csv'):
    """
    Loads the dataset from a CSV file.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return None
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

if __name__ == "__main__":
    # Example usage
    download_data('AAPL', '2015-01-01', '2023-12-31', 'data/AAPL_data.csv')
