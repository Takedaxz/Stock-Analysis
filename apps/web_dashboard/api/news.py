from http.server import BaseHTTPRequestHandler
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient, DESCENDING
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Vercel deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["stock_news_db"]
collection = db["news_data"]

@app.get("/api/news")
def get_news(ticker: str = Query("news")):
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

# Vercel serverless function handler
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api/news'):
            # Extract query parameters
            from urllib.parse import urlparse, parse_qs
            parsed_url = urlparse(self.path)
            params = parse_qs(parsed_url.query)
            ticker = params.get('ticker', ['news'])[0]
            
            # Get news data
            news_data = get_news(ticker)
            
            # Return response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(json.dumps(news_data).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers() 