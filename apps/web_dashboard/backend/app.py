# backend/app.py
from fastapi import FastAPI,Query
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient, DESCENDING
import os
from dotenv import load_dotenv
import yfinance as yf
import ta

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stock-analysis-theta-inky.vercel.app",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["stock_news_db"]
collection = db["news_data"]

@app.get("/news")
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

@app.get("/chart-data")
def get_chart_data(symbol: str = Query("^GSPC")):
    df = yf.download(symbol, period="6mo",interval="1d", progress=False, multi_level_index=False)
    df = df.dropna().reset_index()

    df["EMA5"] = ta.trend.ema_indicator(df["Close"], window=5).bfill()
    df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20).bfill()


    return {
        "symbol": symbol,
        "timestamps": [d.strftime('%Y-%m-%d') for d in df["Date"]],
        "prices": df["Close"].values.tolist(),
        "ema20": df["EMA20"].tolist(),
        "ema5": df["EMA5"].tolist(),
    }