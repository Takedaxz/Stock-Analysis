from http.server import BaseHTTPRequestHandler
from pymongo import MongoClient, DESCENDING
import os
import json
from urllib.parse import urlparse, parse_qs

def get_news(ticker="news"):
    try:
        MONGO_URI = os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
        client = MongoClient(MONGO_URI)
        db = client["stock_news_db"]
        collection = db["news_data"]
        
        # If ticker is a market index, get general news
        if ticker.startswith("^"):
            news = list(
                collection.find({"ticker": "news"}, {"_id": 0})
                .sort([("publish_date", DESCENDING), ("publish_time", DESCENDING)])
                .limit(20)
            )
        else:
            # For individual stocks, search for news specific to that ticker
            news = list(
                collection.find({"ticker": ticker}, {"_id": 0})
                .sort([("publish_date", DESCENDING), ("publish_time", DESCENDING)])
                .limit(20)
            )
            # If no specific news found for the ticker, fall back to general news
            if not news:
                news = list(
                    collection.find({"ticker": "news"}, {"_id": 0})
                    .sort([("publish_date", DESCENDING), ("publish_time", DESCENDING)])
                    .limit(20)
                )
        return news
    except Exception as e:
        return {"error": str(e)}

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse the URL to get query parameters
        parsed_url = urlparse(self.path)
        params = parse_qs(parsed_url.query)
        ticker = params.get('ticker', ['news'])[0]
        
        # Get news data
        news_data = get_news(ticker)
        
        # Set response headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        # Return JSON response
        self.wfile.write(json.dumps(news_data).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers() 