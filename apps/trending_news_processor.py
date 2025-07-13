#!/usr/bin/env python3
"""
Trending News Processor - Using News APIs
Uses NewsAPI and NewsData.io for reliable news fetching
"""

import os
import re
import time
import warnings
from datetime import datetime, timedelta
import pandas as pd
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm
from google.api_core import retry
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configuration
MAX_ARTICLES = 30  # Reduced to avoid rate limits
FINAL_ARTICLES = 20  # Reduced to avoid rate limits (10 articles = 20 API calls max)

def fetch_newsapi_articles():
    """Fetch articles from NewsAPI"""
    print("Fetching articles from NewsAPI...")
    
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        print("NEWSAPI_KEY not found, skipping NewsAPI")
        return []
    
    # Get articles from the last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': 'stock market OR trading OR stocks OR market OR earnings OR fed OR inflation OR interest rates',
        'language': 'en',
        'sortBy': 'popularity',
        'pageSize': MAX_ARTICLES,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'ok':
            articles = []
            for article in data['articles']:
                articles.append({
                    'title': article.get('title', 'No title'),
                    'body_text': article.get('content', article.get('description', '')),
                    'url': article.get('url', ''),
                    'publish_date': article.get('publishedAt', '').split('T')[0] if article.get('publishedAt') else datetime.now().strftime('%Y-%m-%d'),
                    'publish_time': article.get('publishedAt', '').split('T')[1][:5] if article.get('publishedAt') else '00:00',
                    'source': article.get('source', {}).get('name', 'Unknown')
                })
            print(f"Fetched {len(articles)} articles from NewsAPI")
            return articles
        else:
            print(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return []
    except Exception as e:
        print(f"Error fetching from NewsAPI: {e}")
        return []

def fetch_alphavantage_articles():
    """Fetch articles from Alpha Vantage News API"""
    print("Fetching articles from Alpha Vantage...")
    
    api_key = os.getenv("ALPHAVANTAGE_KEY")
    if not api_key:
        print("ALPHAVANTAGE_KEY not found, skipping Alpha Vantage")
        return []
    
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'NEWS_SENTIMENT',
        'topics': 'financial_markets',
        'limit': MAX_ARTICLES,
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'feed' in data:
            articles = []
            for article in data['feed']:
                # Parse the time_published
                time_published = article.get('time_published', '')
                publish_date = datetime.now().strftime('%Y-%m-%d')
                publish_time = '00:00'
                
                if time_published:
                    try:
                        # Alpha Vantage format: 20240101T000000
                        date_str = time_published[:8]  # YYYYMMDD
                        time_str = time_published[9:13]  # HHMM
                        publish_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                        publish_time = f"{time_str[:2]}:{time_str[2:4]}"
                    except:
                        pass
                
                articles.append({
                    'title': article.get('title', 'No title'),
                    'body_text': article.get('summary', ''),
                    'url': article.get('url', ''),
                    'publish_date': publish_date,
                    'publish_time': publish_time,
                    'source': article.get('source', 'Unknown')
                })
            print(f"Fetched {len(articles)} articles from Alpha Vantage")
            return articles
        else:
            print(f"Alpha Vantage error: {data.get('Note', 'Unknown error')}")
            return []
    except Exception as e:
        print(f"Error fetching from Alpha Vantage: {e}")
        return []

def combine_articles():
    """Combine articles from NewsAPI and return top 20 latest"""
    print("Combining articles from News APIs...")
    
    all_articles = []
    
    # Fetch from NewsAPI
    newsapi_articles = fetch_newsapi_articles()
    all_articles.extend(newsapi_articles)
    
    # For now, skip Alpha Vantage due to potential API issues
    alphavantage_articles = fetch_alphavantage_articles()
    all_articles.extend(alphavantage_articles)
    
    # Remove duplicates based on URL
    seen_urls = set()
    unique_articles = []
    
    for article in all_articles:
        url = article.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)
    
    print(f"Total unique articles: {len(unique_articles)}")
    
    # Convert to DataFrame and sort by date/time
    if unique_articles:
        df = pd.DataFrame(unique_articles)
        # Add ticker column
        df['ticker'] = 'news'
        
        # Add impact score based on market-relevant keywords
        market_keywords = [
            'stock market', 'trading', 'stocks', 'market', 'earnings', 'fed', 'inflation', 
            'interest rates', 'dow', 's&p', 'nasdaq', 'trading volume', 'market rally',
            'market correction', 'bull market', 'bear market', 'volatility', 'recession',
            'economic', 'financial', 'investor', 'trading session', 'market close'
        ]
        
        def calculate_impact_score(title, body):
            title_lower = title.lower()
            body_lower = body.lower()
            score = 0
            
            for keyword in market_keywords:
                if keyword in title_lower:
                    score += 2  # Moderate weight for title matches
                if keyword in body_lower:
                    score += 1  # Lower weight for body matches
            
            return score
        
        # Calculate impact score for sorting only (not stored in DB)
        df['temp_impact_score'] = df.apply(lambda row: calculate_impact_score(row['title'], row['body_text']), axis=1)
        
        # Sort by date first (newest), then by impact score
        df = df.sort_values(by=['publish_date', 'publish_time', 'temp_impact_score'], ascending=[False, False, False]).reset_index(drop=True)
        
        # Take only the top 20 latest articles
        df = df.head(FINAL_ARTICLES)
        
        # Remove the temporary impact score column before returning
        df = df.drop(columns=['temp_impact_score'])
        
        print(f"Selected top {len(df)} latest market-relevant articles")
        return df
    else:
        return pd.DataFrame()

def setup_gemini():
    """Setup Gemini API"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    generation_config = genai.GenerationConfig(temperature=0)
    return genai.GenerativeModel("gemini-2.5-flash-preview-04-17", generation_config=generation_config)

def is_retryable(e) -> bool:
    """Check if error is retryable"""
    if retry.if_transient_error(e):
        return True
    elif (isinstance(e, genai.errors.ClientError) and e.code == 429):
        return True
    elif (isinstance(e, genai.errors.ServerError) and e.code == 503):
        return True
    else:
        return False

@retry.Retry(predicate=is_retryable)
def generate_content_with_rate_limit(model, prompt):
    """Generate content with rate limiting"""
    # Add delay to respect rate limits (15 requests per minute = 4 seconds between requests)
    time.sleep(4)  # 4 second delay between requests
    return model.generate_content(prompt).text

def analyze_sentiment(model, df):
    """Analyze sentiment for all articles"""
    prompt = """You are a financial news analyst specializing in stock market impact. Your task is to analyze the provided news article, summarize its core content concisely, determine its sentiment (positive, negative, or neutral), and assess its importance to the specified stock.

Here is the news from stock [STOCK] title and body:
---
[TITLE]
---
[BODY]
---

Please provide your analysis in the following format (Don't forget to make space between the sections as shown):

**Sentiment:**
[Positive / Negative / Neutral]

**Summary:**
[Your concise summary of the article, typically 2-3 sentences.]

**Reasoning for Sentiment:**
[Brief explanation (1-2 sentences) of why you categorized the sentiment as such, referencing key points or tone from the article.]

**Importance to Stock [STOCK]:**
[1-5, where 1 is minimal importance and 5 is very high importance.Answer in 1-5 only, no explanation.] (Answer only in number 1-5)

**Reasoning for Importance:**
[Brief explanation (1-2 sentences) of why you assigned this importance score, referencing specific details from the article that would impact the stock.]"""

    predicted = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing sentiment"):
        current_stock = row.get("ticker", "news")
        
        filled_prompt = prompt.replace("[STOCK]", current_stock)
        filled_prompt = filled_prompt.replace("[TITLE]", row["title"])
        filled_prompt = filled_prompt.replace("[BODY]", row["body_text"])
        
        try:
            response = generate_content_with_rate_limit(model, filled_prompt)
            finalprediction = response.strip()
            if not finalprediction:
                print(f"Row {index}: LLM returned an empty string.")
                predicted.append("LLM_EMPTY_RESPONSE")
            else:
                predicted.append(finalprediction)
                print(f"✅ Processed article {index + 1}/{len(df)}")
        except Exception as e:
            print(f"Row {index}: Error - {e}")
            if "quota" in str(e).lower() or "429" in str(e).lower():
                print("⚠️  Rate limit hit - stopping processing")
                break
            predicted.append("ERROR_UNEXPECTED")
            continue
    
    return predicted

def translate_summaries(model, df):
    """Translate summaries to Thai"""
    prompt = """Translate the following English sentence to Thai. Do not translate proper nouns, company names, product names, abbreviations, or technical terms — keep them in English. Do not provide any explanation, just the translation.
[TEXT]"""
    
    translate = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Translating"):
        filled_prompt = prompt.replace("[TEXT]", row["summary"])
        
        try:
            response = generate_content_with_rate_limit(model, filled_prompt)
            finalprediction = response.strip()
            if not finalprediction:
                print(f"Row {index}: LLM returned an empty string.")
                translate.append("LLM_EMPTY_RESPONSE")
            else:
                translate.append(finalprediction)
                print(f"✅ Translated article {index + 1}/{len(df)}")
        except Exception as e:
            print(f"Row {index}: Error - {e}")
            if "quota" in str(e).lower() or "429" in str(e).lower():
                print("⚠️  Rate limit hit - stopping translation")
                break
            translate.append("ERROR_UNEXPECTED")
            continue
    
    return translate

def setup_mongodb():
    """Setup MongoDB connection"""
    mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")
    if not mongo_connection_string:
        raise ValueError("MONGO_CONNECTION_STRING not found in environment variables")
    
    client = MongoClient(mongo_connection_string)
    db = client['stock_news_db']
    collection = db['news_data']
    
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
    return collection

def main():
    print("="*50)
    print("Starting News API Processor")
    print("="*50)
    
    # Setup APIs
    model = setup_gemini()
    collection = setup_mongodb()
    
    try:
        print("Fetching articles from News APIs...")
        df = combine_articles()
        
        if df.empty:
            print("No articles found from APIs")
            return
        
        print(f"Fetched {len(df)} articles from APIs")
        
        print("Analyzing sentiment...")
        print(f"Processing {len(df)} articles with 4-second delays between API calls...")
        predicted = analyze_sentiment(model, df)
        df["predicted"] = predicted
        print("Sentiment analysis complete")
        
        df["sentiment"] = df["predicted"].apply(lambda x: x.split("\n")[1].strip() if len(x.split("\n")) > 1 else None)
        df["importance"] = df["predicted"].apply(lambda x: x.split("\n")[10].strip() if len(x.split("\n")) > 10 else None)
        df["summary"] = df["predicted"].apply(lambda x: x.split("\n")[4].strip() if len(x.split("\n")) > 4 else None)
        
        df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
        df = df[df['importance'].isin(['1', '2', '3', '4', '5'])]
        
        if df.empty:
            print("No valid sentiment analysis")
            return
        
        print("Translating summaries...")
        print(f"Translating {len(df)} summaries with 4-second delays between API calls...")
        translate = translate_summaries(model, df)
        df["translate"] = translate
        print("Translation complete")
        
        if 'predicted' in df.columns:
            df.drop(columns=['predicted'], inplace=True)
        if 'body_text' in df.columns:
            df.drop(columns=['body_text'], inplace=True)
        
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d %H-%M").strip().replace(' ', '_')
        filename = f"gemini_news_api_{date_time}.csv".lower()
        
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(f"data/processed/{filename}", index=False)
        print("Saved CSV file")
        
        complete_dict = df.to_dict(orient='records')
        result = collection.insert_many(complete_dict, ordered=True)
        print(f"Inserted {len(result.inserted_ids)} documents to MongoDB")
        
        print("="*50)
        print("News API Processing Complete!")
        print("="*50)
        
    except Exception as e:
        print(f"Error processing news: {e}")
        raise

if __name__ == "__main__":
    main() 