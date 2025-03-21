import yfinance as yf
import pandas as pd
import numpy as np


# interval is the frequency of the data
# period is the duration of the data
# auto_adjust is a boolean that adjusts the data for splits and dividends

def get_history (
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1mo", # Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        period="1y",
        auto_adjust=True,
        fields=["Open", "Close", "Volume", "Dividends", "Stock Splits"]
    ) -> pd.DataFrame:
    
    history = yf.Ticker(ticker).history(start=start_date,
                                        end=end_date,
                                        interval=interval, 
                                        period=period,
                                        auto_adjust=auto_adjust)
    return history[fields]


if __name__ == "__main__":
    history = get_history("AAPL", "2019-01-01", "2021-12-31")
    print (history.head())
    print (history.tail())
    print (history.shape)
    print (history.columns)
    print (history.index)
    print (history.describe())
    print (history.info())
    