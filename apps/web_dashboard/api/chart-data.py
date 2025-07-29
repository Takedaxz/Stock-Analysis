from http.server import BaseHTTPRequestHandler
import requests
import ta
import json
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta
import pandas as pd

def get_real_data(symbol="^GSPC"):
    """Fetch real data from Yahoo Finance using requests"""
    try:
        # Calculate date range (6 months ago to now)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        # Format dates for Yahoo Finance
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        # Yahoo Finance API URL
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_timestamp}&period2={end_timestamp}&interval=1d"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            return {"error": "No data available", "symbol": symbol}
        
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        # Convert timestamps to dates
        dates = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]
        
        # Get close prices
        closes = quotes['close']
        
        # Create DataFrame for technical indicators
        df = pd.DataFrame({
            'Date': dates,
            'Close': closes
        })
        
        # Calculate EMAs
        df["EMA5"] = ta.trend.ema_indicator(df["Close"], window=5).bfill()
        df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20).bfill()
        
        return {
            "symbol": symbol,
            "timestamps": dates,
            "prices": df["Close"].tolist(),
            "ema20": df["EMA20"].tolist(),
            "ema5": df["EMA5"].tolist(),
        }
        
    except Exception as e:
        print(f"Error fetching real data: {str(e)}")
        return {"error": str(e), "symbol": symbol}

def get_chart_data(symbol="^GSPC"):
    return get_real_data(symbol)

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