from http.server import BaseHTTPRequestHandler
import yfinance as yf
import ta
import json
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta
import random

def get_mock_data(symbol="^GSPC"):
    """Generate mock chart data for demonstration"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # Generate dates
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    # Generate mock prices (starting around 4000 for S&P 500)
    base_price = 4000 if symbol == "^GSPC" else 150
    prices = []
    for i in range(len(dates)):
        # Add some random variation
        variation = random.uniform(-0.02, 0.02)
        price = base_price * (1 + variation)
        prices.append(round(price, 2))
        base_price = price
    
    # Calculate mock EMAs
    ema5 = []
    ema20 = []
    for i in range(len(prices)):
        if i < 4:
            ema5.append(prices[i])
        else:
            ema5_val = (prices[i] * 0.4) + (ema5[i-1] * 0.6)
            ema5.append(round(ema5_val, 2))
        
        if i < 19:
            ema20.append(prices[i])
        else:
            ema20_val = (prices[i] * 0.05) + (ema20[i-1] * 0.95)
            ema20.append(round(ema20_val, 2))
    
    return {
        "symbol": symbol,
        "timestamps": dates,
        "prices": prices,
        "ema20": ema20,
        "ema5": ema5,
    }

def get_chart_data(symbol="^GSPC"):
    try:
        print(f"Downloading data for symbol: {symbol}")
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
        print(f"Downloaded data shape: {df.shape}")
        
        if df.empty:
            print("No data from yfinance, using mock data")
            return get_mock_data(symbol)
            
        df = df.dropna().reset_index()
        print(f"After dropna shape: {df.shape}")

        df["EMA5"] = ta.trend.ema_indicator(df["Close"], window=5).bfill()
        df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20).bfill()

        return {
            "symbol": symbol,
            "timestamps": [d.strftime('%Y-%m-%d') for d in df["Date"]],
            "prices": df["Close"].values.tolist(),
            "ema20": df["EMA20"].tolist(),
            "ema5": df["EMA5"].tolist(),
        }
    except Exception as e:
        print(f"Error in get_chart_data: {str(e)}")
        print("Using mock data instead")
        return get_mock_data(symbol)

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse the URL to get query parameters
        parsed_url = urlparse(self.path)
        params = parse_qs(parsed_url.query)
        symbol = params.get('symbol', ['^GSPC'])[0]
        
        # Get chart data
        chart_data = get_chart_data(symbol)
        
        # Set response headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        # Return JSON response
        self.wfile.write(json.dumps(chart_data).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers() 