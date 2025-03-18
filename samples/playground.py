import yfinance as yf
import pandas as pd
import datetime

# --- CONFIG ---
STOCK_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Start small, scale later
START_DATE = '2010-01-01'
END_DATE = '2024-01-01'
SP500_TICKER = '^GSPC'
FED_RATE_TICKER = '^IRX'  # 13-week Treasury Bill rate (proxy for Fed Rate)

# --- Fetch Historical Prices ---
def get_stock_data(ticker):
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, interval='1mo')
        if data.empty:
            print(f"Warning: No data fetched for {ticker}")
            return None

        if 'Close' not in data.columns:
            print(f"Warning: 'Close' not found for {ticker}. Available columns: {list(data.columns)}")
            return None

        data = data[['Close']].rename(columns={'Close': ticker})
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# --- Fetch All Data ---
def fetch_all_data():
    price_data = []
    for ticker in STOCK_TICKERS + [SP500_TICKER]:
        df = get_stock_data(ticker)
        if df is not None:
            price_data.append(df)

    if not price_data:
        raise ValueError("No valid price data fetched.")

    merged_prices = pd.concat(price_data, axis=1)
    merged_prices = merged_prices.dropna()

    # Fetch Fed Rate (monthly interval)
    fed_rate = yf.download(FED_RATE_TICKER, start=START_DATE, end=END_DATE, interval='1mo')
    if 'Close' not in fed_rate.columns:
        raise ValueError(f"Fed Rate data missing 'Close'. Available columns: {list(fed_rate.columns)}")
    fed_rate = fed_rate[['Close']].rename(columns={'Close': 'FedRate'})
    fed_rate = fed_rate / 100  # Convert to decimal rate

    merged = merged_prices.merge(fed_rate, left_index=True, right_index=True)
    return merged

# --- Calculate 12M Returns ---
def calculate_returns(df):
    returns = df.pct_change(periods=12)
    returns = returns.shift(-12)  # Align future return with current date
    return returns

# --- Create Label ---
def create_labels(stock_returns, sp500_returns_df, fed_rate):
    aligned_labels = []
    for ticker in STOCK_TICKERS:
        sr, sp500 = stock_returns[ticker].align(sp500_returns_df['SP500'], join='inner')
        sr, fed = sr.align(fed_rate['FedRate'], join='inner')
        beat_sp500 = sr >= sp500
        beat_fed = sr >= fed
        label_series = (beat_sp500 & beat_fed).astype(int)
        label_series.name = ticker
        aligned_labels.append(label_series)

    labels_df = pd.concat(aligned_labels, axis=1)
    return labels_df

# --- Main ---
if __name__ == "__main__":
    merged_data = fetch_all_data()

    # Split prices
    price_df = merged_data[STOCK_TICKERS]
    sp500_df = merged_data[[SP500_TICKER]]
    fed_df = merged_data[['FedRate']]

    stock_returns = calculate_returns(price_df)
    sp500_returns = calculate_returns(sp500_df).rename(columns={SP500_TICKER: 'SP500'})

    labels = create_labels(stock_returns, sp500_returns, fed_df)

    # Preview
    print("Sample Data:")
    print(labels.head())

    # Save for model training
    stock_returns.to_csv("stock_returns.csv")
    labels.to_csv("stock_labels.csv")
