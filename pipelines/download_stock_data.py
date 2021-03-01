import yfinance as yf

def main():
    data = yf.download("SPY AAPL", start="2017-01-01", end="2017-04-30")
    print(data)

if __name__ == '__main__':
    main()