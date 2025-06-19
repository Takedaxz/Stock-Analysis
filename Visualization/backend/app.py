# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient, DESCENDING
import os
from dotenv import load_dotenv

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stock-analysis-theta-inky.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["stock_news_db"]
collection = db["news_data"]

@app.get("/news")
def get_news():
    news = list(
        collection.find({"ticker": "news"}, {"_id": 0})
        .sort([("publish_date", DESCENDING), ("publish_time", DESCENDING)])
        .limit(20)
    )
    return news