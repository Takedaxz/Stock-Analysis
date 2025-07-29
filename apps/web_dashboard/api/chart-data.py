from http.server import BaseHTTPRequestHandler
import yfinance as yf
import ta
import json
from urllib.parse import urlparse, parse_qs

def get_chart_data(symbol="^GSPC"):
    try:
        print(f"Downloading data for symbol: {symbol}")
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
        print(f"Downloaded data shape: {df.shape}")
        
        if df.empty:
            return {"error": "No data received from yfinance", "symbol": symbol}
            
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
        return {"error": str(e), "symbol": symbol}

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