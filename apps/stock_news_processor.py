#!/usr/bin/env python3
"""
Stock News Processor - Using News APIs
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
import google.generativeai as genai
from tqdm import tqdm
from google.api_core import retry
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from difflib import SequenceMatcher
import hashlib

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configuration
COMPANIES = {
    "Tesla": "TSLA",
    "NVIDIA": "NVDA", 
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Meta": "META",
    "Netflix": "NFLX",
    "AMD": "AMD",
}

# Process all companies or specify one
TARGET_COMPANIES = []  # Will be populated by user input

MAX_ARTICLES = 30
FINAL_ARTICLES = 20

# Duplicate detection settings
SIMILARITY_THRESHOLD = 0.75  # 75% similarity threshold for content
TITLE_SIMILARITY_THRESHOLD = 0.45  # 45% similarity threshold for titles (lowered for better detection)
URL_SIMILARITY_THRESHOLD = 0.90  # 90% similarity threshold for URLs
ENABLE_DETAILED_LOGGING = True  # Enable detailed logging of duplicate detection

def clean_text(text):
    """Clean and normalize text for comparison"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove numbers (optional - uncomment if you want to ignore numbers)
    # text = re.sub(r'\d+', '', text)
    
    # Remove common words that don't add meaning
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 
        'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your', 'he', 'she', 
        'his', 'her', 'him', 'as', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
        'after', 'above', 'below', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
    }
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words).strip()

def calculate_similarity(text1, text2):
    """Calculate similarity between two texts using SequenceMatcher"""
    if not text1 or not text2:
        return 0.0
    
    clean_text1 = clean_text(text1)
    clean_text2 = clean_text(text2)
    
    if not clean_text1 or not clean_text2:
        return 0.0
    
    return SequenceMatcher(None, clean_text1, clean_text2).ratio()

def is_duplicate_article(article1, article2):
    """Check if two articles are duplicates based on URL and content similarity"""
    # Check URL similarity first (fast check)
    url1 = article1.get('url', '').lower()
    url2 = article2.get('url', '').lower()
    
    # If URLs are identical or very similar, it's a duplicate
    if url1 and url2:
        if url1 == url2:
            return True
        # Check if URLs are similar (same domain, similar path)
        if calculate_similarity(url1, url2) > URL_SIMILARITY_THRESHOLD:
            return True
    
    # Check title similarity
    title1 = article1.get('title', '')
    title2 = article2.get('title', '')
    
    if title1 and title2:
        title_similarity = calculate_similarity(title1, title2)
        if title_similarity > TITLE_SIMILARITY_THRESHOLD:
            return True
    
    # Check content similarity
    body1 = article1.get('body_text', '')
    body2 = article2.get('body_text', '')
    
    if body1 and body2:
        content_similarity = calculate_similarity(body1, body2)
        if content_similarity > SIMILARITY_THRESHOLD:
            return True
    
    return False

def remove_duplicates(articles):
    """Remove duplicate articles based on URL and content similarity"""
    if not articles:
        return []
    
    unique_articles = []
    seen_urls = set()
    duplicate_count = 0
    
    for i, article in enumerate(articles):
        url = article.get('url', '').lower()
        title = article.get('title', '')
        
        # Skip if URL is already seen
        if url in seen_urls:
            duplicate_count += 1
            if ENABLE_DETAILED_LOGGING:
                print(f"  Duplicate found (URL): {title[:50]}...")
            continue
        
        # Check if this article is a duplicate of any existing unique article
        is_duplicate = False
        duplicate_reason = ""
        
        for j, unique_article in enumerate(unique_articles):
            if is_duplicate_article(article, unique_article):
                is_duplicate = True
                # Determine the reason for duplication
                url1 = article.get('url', '').lower()
                url2 = unique_article.get('url', '').lower()
                title1 = article.get('title', '')
                title2 = unique_article.get('title', '')
                body1 = article.get('body_text', '')
                body2 = unique_article.get('body_text', '')
                
                if url1 and url2 and calculate_similarity(url1, url2) > URL_SIMILARITY_THRESHOLD:
                    duplicate_reason = "similar URL"
                elif title1 and title2 and calculate_similarity(title1, title2) > TITLE_SIMILARITY_THRESHOLD:
                    duplicate_reason = "similar title"
                elif body1 and body2 and calculate_similarity(body1, body2) > SIMILARITY_THRESHOLD:
                    duplicate_reason = "similar content"
                else:
                    duplicate_reason = "multiple factors"
                
                break
        
        if is_duplicate:
            duplicate_count += 1
            if ENABLE_DETAILED_LOGGING:
                print(f"  Duplicate found ({duplicate_reason}): {title[:50]}...")
        else:
            unique_articles.append(article)
            if url:
                seen_urls.add(url)
    
    if duplicate_count > 0:
        print(f"  Removed {duplicate_count} duplicate articles")
    
    return unique_articles

def test_duplicate_detection():
    """Test function to demonstrate duplicate detection functionality"""
    print("Testing duplicate detection system...")
    
    # Sample articles for testing
    test_articles = [
        {
            'title': 'Tesla Reports Strong Q3 Earnings',
            'body_text': 'Tesla Inc. reported strong third-quarter earnings today, beating analyst expectations.',
            'url': 'https://example.com/tesla-earnings-2024',
            'source': 'Financial Times'
        },
        {
            'title': 'Tesla Q3 Earnings Beat Expectations',
            'body_text': 'Tesla Inc. announced third-quarter results that exceeded Wall Street estimates.',
            'url': 'https://different-site.com/tesla-q3-results',
            'source': 'Reuters'
        },
        {
            'title': 'Apple iPhone Sales Decline',
            'body_text': 'Apple Inc. reported declining iPhone sales in the latest quarter.',
            'url': 'https://example.com/apple-iphone-sales',
            'source': 'Bloomberg'
        },
        {
            'title': 'Tesla Reports Strong Q3 Earnings',
            'body_text': 'Tesla Inc. reported strong third-quarter earnings today, beating analyst expectations.',
            'url': 'https://example.com/tesla-earnings-2024',
            'source': 'Financial Times'
        }
    ]
    
    print(f"Original articles: {len(test_articles)}")
    unique_articles = remove_duplicates(test_articles)
    print(f"After duplicate removal: {len(unique_articles)}")
    
    print("\nUnique articles:")
    for i, article in enumerate(unique_articles, 1):
        print(f"{i}. {article['title']} - {article['source']}")

def fetch_newsapi_articles(ticker):
    """Fetch articles from NewsAPI for specific ticker"""
    print(f"Fetching articles from NewsAPI for {ticker}...")
    
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        print("NEWSAPI_KEY not found, skipping NewsAPI")
        return []
    
    # Get articles from the last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': f'{ticker} OR "{COMPANIES[ticker]}"',
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
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'ticker': ticker
                })
            print(f"Fetched {len(articles)} articles from NewsAPI for {ticker}")
            return articles
        else:
            print(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return []
    except Exception as e:
        print(f"Error fetching from NewsAPI: {e}")
        return []

def fetch_alphavantage_articles(ticker):
    """Fetch articles from Alpha Vantage News API for specific ticker"""
    print(f"Fetching articles from Alpha Vantage for {ticker}...")
    
    api_key = os.getenv("ALPHAVANTAGE_KEY")
    if not api_key:
        print("ALPHAVANTAGE_KEY not found, skipping Alpha Vantage")
        return []
    
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': COMPANIES[ticker],
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
                    'source': article.get('source', 'Unknown'),
                    'ticker': ticker
                })
            print(f"Fetched {len(articles)} articles from Alpha Vantage for {ticker}")
            return articles
        else:
            print(f"Alpha Vantage error: {data.get('Note', 'Unknown error')}")
            return []
    except Exception as e:
        print(f"Error fetching from Alpha Vantage: {e}")
        return []

def fetch_polygon_articles(ticker):
    """Fetch articles from Polygon.io News API for specific ticker"""
    print(f"Fetching articles from Polygon.io for {ticker}...")
    
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("POLYGON_API_KEY not found, skipping Polygon.io")
        return []
    
    # Get articles from the last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        'ticker': COMPANIES[ticker],
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
                    'source': article.get('publisher', {}).get('name', 'Unknown'),
                    'ticker': ticker
                })
            print(f"Fetched {len(articles)} articles from Polygon.io for {ticker}")
            return articles
        else:
            print(f"Polygon.io error: No results found for {ticker}")
            return []
    except Exception as e:
        print(f"Error fetching from Polygon.io: {e}")
        return []

def combine_articles_for_ticker(ticker):
    """Combine articles from News APIs for specific ticker and return top 20 latest"""
    print(f"Combining articles from News APIs for {ticker}...")
    
    all_articles = []
    
    # Fetch from NewsAPI
    newsapi_articles = fetch_newsapi_articles(ticker)
    all_articles.extend(newsapi_articles)
    
    # Fetch from Alpha Vantage
    alphavantage_articles = fetch_alphavantage_articles(ticker)
    all_articles.extend(alphavantage_articles)
    
    # Fetch from Polygon.io
    polygon_articles = fetch_polygon_articles(ticker)
    all_articles.extend(polygon_articles)
    
    print(f"Total articles fetched from all APIs for {ticker}: {len(all_articles)}")
    
    # Remove duplicates based on URL and content similarity
    unique_articles = remove_duplicates(all_articles)
    
    print(f"Total unique articles after duplicate removal for {ticker}: {len(unique_articles)}")
    print(f"Removed {len(all_articles) - len(unique_articles)} duplicate articles")
    
    # Convert to DataFrame and sort by date/time
    if unique_articles:
        df = pd.DataFrame(unique_articles)
        
        # Sort by date first (newest), then by time
        df = df.sort_values(by=['publish_date', 'publish_time'], ascending=[False, False]).reset_index(drop=True)
        
        # Take only the top 20 latest articles
        df = df.head(FINAL_ARTICLES)
        
        print(f"Selected top {len(df)} latest articles for {ticker}")
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
    return genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config)

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

def analyze_sentiment_and_translate(model, df, company_name):
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
        current_stock = COMPANIES[company_name]
        
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
    print("[DEBUG] Setting up MongoDB connection")
    mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")
    if not mongo_connection_string:
        raise ValueError("MONGO_CONNECTION_STRING not found in environment variables")
    
    client = MongoClient(mongo_connection_string)
    db = client['stock_news_db']
    collection = db['news_data']
    
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
    print("[DEBUG] MongoDB connection successful!")
    return collection

def get_user_input():
    """Get stock selection from user"""
    print("\n" + "="*60)
    print("STOCK NEWS PROCESSOR")
    print("="*60)
    print("\nAvailable stocks:")
    
    for i, (company_name, ticker) in enumerate(COMPANIES.items(), 1):
        print(f"{i:2d}. {company_name} ({ticker})")
    
    print(f"{len(COMPANIES) + 1:2d}. Process ALL stocks")
    print(f"{len(COMPANIES) + 2:2d}. Test Duplicate Detection")
    print(f"{len(COMPANIES) + 3:2d}. Exit")
    
    while True:
        try:
            choice = input(f"\nEnter your choice (1-{len(COMPANIES) + 2}): ").strip()
            choice_num = int(choice)
            
            selected_companies = []
            if choice_num == len(COMPANIES) + 1:
                # Process all stocks
                selected_companies = list(COMPANIES.keys())
            elif choice_num == len(COMPANIES) + 2:
                # Test duplicate detection
                test_duplicate_detection()
                return []
            elif choice_num == len(COMPANIES) + 3:
                # Exit
                print("Exiting...")
                return []
            elif 1 <= choice_num <= len(COMPANIES):
                # Process specific stock
                company_name = list(COMPANIES.keys())[choice_num - 1]
                selected_companies = [company_name]
            else:
                print(f"Please enter a number between 1 and {len(COMPANIES) + 3}")
                continue
            
            return selected_companies
            
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            return []

def main():
    print("="*50)
    print("Starting Stock News Processor")
    print("="*50)
    
    # Get user input for stock selection
    selected_companies = get_user_input()
    
    if not selected_companies:
        print("No companies selected. Exiting.")
        return
    
    print(f"\nSelected companies: {', '.join(selected_companies)}")
    
    # Setup APIs
    model = setup_gemini()
    collection = setup_mongodb()
    
    all_results = []
    
    for company_name in selected_companies:
        print(f"\n{'='*50}")
        print(f"Processing {company_name}")
        print(f"{'='*50}")
        
        try:
            print(f"Fetching articles for {company_name}...")
            df = combine_articles_for_ticker(company_name)
            
            if df.empty:
                print(f"No articles found for {company_name}")
                continue
            
            print(f"Fetched {len(df)} articles for {company_name}")
            
            print("Analyzing sentiment and translating...")
            print(f"Processing {len(df)} articles with 4-second delays between API calls...")
            results = analyze_sentiment_and_translate(model, df, company_name)
            df["results"] = results
            print("Analysis and translation complete")
            
            df["sentiment"] = df["results"].apply(lambda x: x.split("\n")[1].strip() if len(x.split("\n")) > 1 else None)
            df["importance"] = df["results"].apply(lambda x: x.split("\n")[13].strip() if len(x.split("\n")) > 13 else None)
            df["summary"] = df["results"].apply(lambda x: x.split("\n")[4].strip() if len(x.split("\n")) > 4 else None)
            df["translate"] = df["results"].apply(lambda x: x.split("\n")[7].strip() if len(x.split("\n")) > 7 else None)
            
            df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
            df = df[df['importance'].isin(['1', '2', '3', '4', '5'])]
            df['ticker'] = COMPANIES[company_name]
            if df.empty:
                print(f"No valid sentiment analysis for {company_name}")
                continue
            
            if 'results' in df.columns:
                df.drop(columns=['results'], inplace=True)
            if 'body_text' in df.columns:
                df.drop(columns=['body_text'], inplace=True)
            
            now = datetime.now()
            date_time = now.strftime("%Y-%m-%d %H-%M").strip().replace(' ', '_')
            ticker = COMPANIES[company_name]
            filename = f"gemini_{ticker}_{date_time}.csv".lower()
            
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv(f"data/processed/{filename}", index=False)
            print("Saved CSV file")
            
            complete_dict = df.to_dict(orient='records')
            result = collection.insert_many(complete_dict, ordered=True)
            print(f"Inserted {len(result.inserted_ids)} documents to MongoDB")
            
            all_results.append(df)
            
        except Exception as e:
            print(f"Error processing {company_name}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print(f"Processing complete! Processed {len(all_results)} companies")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 