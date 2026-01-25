#!/usr/bin/env python3
"""
Trending News Processor - Using News APIs
Uses NewsAPI, Alpha Vantage, and Polygon.io for reliable news fetching
"""

import os
import re
import time
import warnings
from datetime import datetime, timedelta
import pandas as pd
import requests
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm
from google.api_core import retry
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configuration
MAX_ARTICLES = 30
FINAL_ARTICLES = 20

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

def fetch_polygon_articles():
    """Fetch articles from Polygon.io News API"""
    print("Fetching articles from Polygon.io...")
    
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("POLYGON_API_KEY not found, skipping Polygon.io")
        return []
    
    # Get articles from the last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        'published_utc.gte': start_date.strftime('%Y-%m-%d'),
        'published_utc.lte': end_date.strftime('%Y-%m-%d'),
        'order': 'desc',
        'sort': 'published_utc',
        'limit': MAX_ARTICLES,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data and data['results']:
            articles = []
            for article in data['results']:
                # Parse the published_utc
                published_utc = article.get('published_utc', '')
                publish_date = datetime.now().strftime('%Y-%m-%d')
                publish_time = '00:00'
                
                if published_utc:
                    try:
                        # Polygon format: 2024-01-01T00:00:00Z
                        date_part = published_utc.split('T')[0]
                        time_part = published_utc.split('T')[1][:5]  # HH:MM
                        publish_date = date_part
                        publish_time = time_part
                    except:
                        pass
                
                # Combine description and insights for body text
                body_text = article.get('description', '')
                if article.get('insights'):
                    insights_text = ' '.join([insight.get('text', '') for insight in article['insights']])
                    if insights_text:
                        body_text += f" {insights_text}"
                
                articles.append({
                    'title': article.get('title', 'No title'),
                    'body_text': body_text,
                    'url': article.get('article_url', ''),
                    'publish_date': publish_date,
                    'publish_time': publish_time,
                    'source': article.get('publisher', {}).get('name', 'Unknown')
                })
            print(f"Fetched {len(articles)} articles from Polygon.io")
            return articles
        else:
            print(f"Polygon.io error: No results found")
            return []
    except Exception as e:
        print(f"Error fetching from Polygon.io: {e}")
        return []

def combine_articles():
    """Combine articles from News APIs and return top 20 latest"""
    print("Combining articles from News APIs...")
    
    all_articles = []
    
    # Fetch from NewsAPI
    newsapi_articles = fetch_newsapi_articles()
    all_articles.extend(newsapi_articles)
    
    # Fetch from Alpha Vantage
    alphavantage_articles = fetch_alphavantage_articles()
    all_articles.extend(alphavantage_articles)
    
    # Fetch from Polygon.io
    polygon_articles = fetch_polygon_articles()
    all_articles.extend(polygon_articles)
    
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
        
        # Enhanced impact score based on market-relevant keywords with different weights
        market_keywords = {
            # High priority - US market specific
            's&p 500': 5, 's&p500': 5, 'sp500': 5, 'spx': 5, 'spy': 5,
            'dow jones': 5, 'dow': 4, 'djia': 4,
            'nasdaq': 4, 'nasdaq 100': 4, 'qqq': 4,
            'us stock market': 5, 'american stock market': 5, 'wall street': 4,
            
            # Market indices and ETFs
            'vix': 4, 'volatility index': 4, 'fear index': 3,
            'russell 2000': 3, 'russell 3000': 3, 'iwm': 3,
            'vanguard': 3, 'blackrock': 3, 'state street': 3,
            
            # Major US companies (FAANG, etc.)
            'apple': 3, 'aapl': 3, 'microsoft': 3, 'msft': 3, 'google': 3, 'googl': 3,
            'amazon': 3, 'amzn': 3, 'tesla': 3, 'tsla': 3, 'nvidia': 3, 'nvda': 3,
            'meta': 3, 'facebook': 3, 'fb': 3, 'netflix': 3, 'nflx': 3,
            'berkshire hathaway': 3, 'brk': 3, 'jpmorgan': 3, 'jpm': 3,
            
            # Financial terms
            'stock market': 4, 'trading': 3, 'stocks': 3, 'market': 3,
            'earnings': 4, 'earnings report': 4, 'quarterly earnings': 4,
            'fed': 4, 'federal reserve': 4, 'jerome powell': 4,
            'inflation': 4, 'cpi': 4, 'ppi': 3, 'interest rates': 4,
            'trading volume': 3, 'market rally': 3, 'market correction': 3,
            'bull market': 3, 'bear market': 3, 'volatility': 3,
            'recession': 4, 'economic': 3, 'financial': 3,
            'investor': 3, 'trading session': 3, 'market close': 3,
            
            # Sector specific
            'technology sector': 3, 'tech stocks': 3, 'financial sector': 3,
            'healthcare sector': 3, 'energy sector': 3, 'consumer staples': 3,
            'consumer discretionary': 3, 'utilities': 3, 'real estate': 3,
            
            # Market events
            'market crash': 4, 'market selloff': 3, 'market bounce': 3,
            'market recovery': 3, 'market volatility': 3, 'market uncertainty': 3,
            'market sentiment': 3, 'market momentum': 3, 'market trend': 3,
            
            # Economic indicators
            'gdp': 3, 'unemployment': 3, 'jobs report': 3, 'non-farm payrolls': 3,
            'retail sales': 3, 'housing market': 3, 'mortgage rates': 3,
            'consumer confidence': 3, 'manufacturing': 3, 'services': 3,
            
            # Trading terms
            'day trading': 3, 'swing trading': 3, 'options': 3, 'futures': 3,
            'etf': 3, 'mutual fund': 3, 'portfolio': 3, 'diversification': 3,
            'risk management': 3, 'stop loss': 3, 'take profit': 3
        }
        
        def calculate_impact_score(title, body):
            title_lower = title.lower()
            body_lower = body.lower()
            score = 0
            
            # Check for keyword matches with weighted scoring
            for keyword, weight in market_keywords.items():
                if keyword in title_lower:
                    score += weight * 2  # Double weight for title matches
                if keyword in body_lower:
                    score += weight  # Normal weight for body matches
            
            # Bonus for US market focus
            us_market_indicators = ['us', 'united states', 'american', 'wall street', 'new york']
            for indicator in us_market_indicators:
                if indicator in title_lower:
                    score += 2
                if indicator in body_lower:
                    score += 1
            
            # Bonus for recent market events
            recent_events = ['today', 'yesterday', 'this week', 'this month', 'latest', 'breaking']
            for event in recent_events:
                if event in title_lower:
                    score += 1
            
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
    """Setup Gemini API using the new google-genai SDK"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    return genai.Client(api_key=api_key)

def is_retryable(e) -> bool:
    """Check if error is retryable for the new SDK"""
    error_str = str(e).lower()
    if "429" in error_str or "quota" in error_str:
        return True
    if "503" in error_str or "unavailable" in error_str:
        return True
    return False

@retry.Retry(predicate=is_retryable)
def generate_content_with_rate_limit(client, prompt):
    """Generate content with rate limiting using the new SDK"""
    # Add delay to respect rate limits
    time.sleep(4)
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config={'temperature': 0}
    )
    return response.text

def analyze_sentiment_and_translate(model, df):
    """Analyze sentiment and translate summary for all articles in one prompt"""
    prompt = """You are a financial news analyst specializing in stock market impact. Your task is to analyze the provided news article, summarize its core content concisely, determine its sentiment (positive, negative, or neutral), assess its importance to the specified stock, and translate the summary to Thai.

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

**Summary in Thai:**
[Thai translation of the summary. Do not translate proper nouns, company names, product names, abbreviations, or technical terms — keep them in English.]

**Reasoning for Sentiment:**
[Brief explanation (1-2 sentences) of why you categorized the sentiment as such, referencing key points or tone from the article.]

**Importance to Stock [STOCK]:**
[1-5, where 1 is minimal importance and 5 is very high importance. Answer in 1-5 only, no explanation.]

**Reasoning for Importance:**
[Brief explanation (1-2 sentences) of why you assigned this importance score, referencing specific details from the article that would impact the stock.]"""

    results = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing sentiment and translating"):
        current_stock = row.get("ticker", "news")
        
        filled_prompt = prompt.replace("[STOCK]", current_stock)
        filled_prompt = filled_prompt.replace("[TITLE]", row["title"])
        filled_prompt = filled_prompt.replace("[BODY]", row["body_text"])
        
        try:
            response = generate_content_with_rate_limit(model, filled_prompt)
            finalresult = response.strip()
            if not finalresult:
                print(f"Row {index}: LLM returned an empty string.")
                results.append("LLM_EMPTY_RESPONSE")
            else:
                results.append(finalresult)
                print(f"✅ Processed article {index + 1}/{len(df)}")
        except Exception as e:
            print(f"Row {index}: Error - {e}")
            if "quota" in str(e).lower() or "429" in str(e).lower():
                print("⚠️  Rate limit hit - stopping processing")
                break
            results.append("ERROR_UNEXPECTED")
            continue
    
    return results



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
        
        print("Analyzing sentiment and translating...")
        print(f"Processing {len(df)} articles with 4-second delays between API calls...")
        results = analyze_sentiment_and_translate(model, df)
        df["results"] = results
        print("Analysis and translation complete")
        
        df["sentiment"] = df["results"].apply(lambda x: x.split("\n")[1].strip() if len(x.split("\n")) > 1 else None)
        df["importance"] = df["results"].apply(lambda x: x.split("\n")[13].strip() if len(x.split("\n")) > 13 else None)
        df["summary"] = df["results"].apply(lambda x: x.split("\n")[4].strip() if len(x.split("\n")) > 4 else None)
        df["translate"] = df["results"].apply(lambda x: x.split("\n")[7].strip() if len(x.split("\n")) > 7 else None)
        
        df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
        df = df[df['importance'].isin(['1', '2', '3', '4', '5'])]
        
        if df.empty:
            print("No valid sentiment analysis")
            return
        
        if 'results' in df.columns:
            df.drop(columns=['results'], inplace=True)
        if 'body_text' in df.columns:
            df.drop(columns=['body_text'], inplace=True)
        
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d %H-%M").strip().replace(' ', '_')
        filename = f"gemini_news_{date_time}.csv".lower()
        
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