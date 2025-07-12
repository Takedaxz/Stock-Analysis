#!/usr/bin/env python3
"""
Stock News Processor - Automated script version
Converts Jupyter notebook to standalone Python script for automation
"""

import os
import re
import time
import random
import asyncio
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import nest_asyncio
import cloudscraper
from htmldate import find_date
from bs4 import BeautifulSoup
from newspaper import Article
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
COMPANIES = {
    "Tesla": "tesla-motors",
    "NVIDIA": "nvidia-corp", 
    "Apple": "apple-computer-inc",
    "Microsoft": "microsoft-corp",
    "Amazon": "amazon-com-inc",
    "Google": "google-inc",
    "Meta": "facebook-inc",
    "Netflix": "netflix,-inc.",
    "AMD": "adv-micro-device",
}

# Process all companies or specify one
TARGET_COMPANIES = ["Tesla"]  # TEMP: Only Tesla for debug

MAX_PAGE = 1  # TEMP: Only 1 page for debug
MAX_WORKERS = 50
MAX_RETRIES = 8

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
    "Referer": "https://www.investing.com/",
    "DNT": "1"
}

def fetch_page(company, page):
    print(f"[DEBUG] Fetching page {page} for {company}")
    global ticker
    url = f"https://www.investing.com/equities/{company}-news/{page}"
    
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'darwin',
            'mobile': False
        },
        delay=2
    )
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = scraper.get(url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            h1_tag = soup.find('h1', class_='mb-2.5')
            full_text = h1_tag.text.strip()
            match = re.search(r'\(([^)]+)\)', full_text)
            ticker = match.group(1)
                    
            anchors = soup.select(
                'ul[data-test="news-list"] '
                'li article a[data-test="article-title-link"]'
            )
            return [a["href"] for a in anchors if a.has_attr("href")]
        except Exception as e:
            if attempt < MAX_RETRIES:
                backoff = 2 ** (attempt - 1) + random.random()
                time.sleep(backoff)
            else:
                print(f"Page {page} failed after {MAX_RETRIES}: {e}")
    return []

def robust_scrape(company):
    print(f"[DEBUG] Starting robust_scrape for {company}")
    first = fetch_page(company, 1)
    PER_PAGE = len(first)
    if PER_PAGE == 0:
        raise RuntimeError(f"Failed to fetch the first page for {company}. Please check headers or cookies and try again.")
    print(f"Detected {PER_PAGE} links per page, expecting {PER_PAGE * MAX_PAGE} total")

    results = {1: first}
    pages = list(range(2, MAX_PAGE + 1))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_page, company, p): p for p in pages}
        for fut in as_completed(futures):
            p = futures[fut]
            results[p] = fut.result()

        for round in range(1, MAX_RETRIES + 1):
            bad = [p for p, links in results.items() if len(links) != PER_PAGE]
            if not bad:
                print(f"All pages OK after {round-1} retries")
                break
            print(f"Retry round {round} for pages: {bad}")
            futures = {pool.submit(fetch_page, company, p): p for p in bad}
            for fut in as_completed(futures):
                p = futures[fut]
                results[p] = fut.result()
        else:
            print("Retry limit reached; some pages may still be incomplete.")

    total_fetched = sum(len(links) for links in results.values())
    expected = PER_PAGE * MAX_PAGE
    print(f"Total links fetched (including duplicates): {total_fetched} (expected {expected})")

    all_links = set(link for links in results.values() for link in links)
    print(f"Final: got {len(all_links)} unique URLs (expected {expected})")
    print(f"[DEBUG] robust_scrape complete for {company}")
    return list(all_links)

def is_placeholder(html: str) -> bool:
    """Check if HTML is a placeholder page"""
    lower = html.lower() if html else ""
    return (
        'temporarily down for maintenance' in lower
        or 'just a moment' in lower
        or "we're temporarily down" in lower
    )

def safe_find_datetime(url, html_content=None):
    """Safely extract datetime from URL or HTML content"""
    try:
        dt = find_date(url)
        if dt:
            return dt, "00:00"
    except:
        pass
    
    if html_content:
        m = re.search(r"(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{1,2}:\d{2}\s*(?:AM|PM))", html_content)
        if m:
            ds, ts = m.groups()
            try:
                dt = datetime.strptime(f"{ds}, {ts}", "%m/%d/%Y, %I:%M %p")
                return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
            except:
                pass
        
        m = re.search(r"(\d{2}/\d{2}/\d{4}),\s*(\d{2}:\d{2})", html_content)
        if m:
            ds, ts = m.groups()
            for fmt in ("%d/%m/%Y, %H:%M", "%m/%d/%Y, %H:%M"):
                try:
                    dt = datetime.strptime(f"{ds}, {ts}", fmt)
                    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
                except:
                    continue
    
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M")

def fetch_html(url, idx, total):
    """Fetch HTML content with retries"""
    scraper = cloudscraper.create_scraper()
    MAX_FETCH_RETRIES = 5
    RETRY_DELAY = 1
    
    for attempt in range(1, MAX_FETCH_RETRIES + 1):
        try:
            resp = scraper.get(url, timeout=30)
            html = resp.text
            if is_placeholder(html):
                raise RuntimeError('Placeholder')
                
            print(f"[Fetch][{idx}/{total}][ok]")
            return url, html
            
        except Exception:
            print(f"[Fetch][{idx}/{total}][retry {attempt}]")
            if attempt < MAX_FETCH_RETRIES:
                time.sleep(RETRY_DELAY)
                
    print(f"[Fetch error] {idx}/{total}: failed after {MAX_FETCH_RETRIES} retries")
    return url, None

def process_article(arg):
    print(f"[DEBUG] Processing article: {arg[0]}")
    url, html = arg
    if not html:
        return None
        
    art = Article(url)
    art.set_html(html)
    
    try:
        art.parse()
    except:
        return None
        
    text = art.text or ""
    title = (art.title or "").strip() or "No title"
    
    date, tm = safe_find_datetime(url, html)
    
    return {'ticker': ticker, 'publish_date': date, 'publish_time': tm,
             'title': title, 'body_text': text, 'url': url}

async def scrape_all(urls):
    print(f"[DEBUG] scrape_all: {len(urls)} URLs")
    total = len(urls)
    loop = asyncio.get_event_loop()
    
    FETCH_WORKERS = min(32, os.cpu_count() * 4)
    PROCESS_WORKERS = os.cpu_count() or 4
    
    # Phase 1: Fetch HTML content
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as fetch_pool:
        fetch_tasks = [loop.run_in_executor(fetch_pool, fetch_html, u, i+1, total)
                       for i, u in enumerate(urls)]
        fetched = await asyncio.gather(*fetch_tasks)

    # Phase 2: Process articles
    records = []
    with ThreadPoolExecutor(max_workers=PROCESS_WORKERS) as proc_pool:
        futures = {
            proc_pool.submit(process_article, fr): fr[0]
            for fr in fetched if fr[1]
        }
        
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            print(f"[Process][{i}/{total}] {futures[fut]}")
            if res:
                records.append(res)
                
    print(f"[DEBUG] scrape_all complete")
    return pd.DataFrame(records)

def setup_gemini():
    print("[DEBUG] Setting up Gemini API")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    generation_config = genai.GenerationConfig(temperature=0)
    print("[DEBUG] Gemini API setup complete")
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
        except Exception as e:
            print(f"Row {index}: Error - {e}")
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
        except Exception as e:
            print(f"Row {index}: Error - {e}")
            translate.append("ERROR_UNEXPECTED")
            continue
    
    return translate

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

def main():
    print("[DEBUG] Main started")
    nest_asyncio.apply()
    
    # Setup APIs
    model = setup_gemini()
    collection = setup_mongodb()
    
    all_results = []
    
    for company_name in TARGET_COMPANIES:
        print(f"\n{'='*50}")
        print(f"Processing {company_name}")
        print(f"{'='*50}")
        
        try:
            print(f"[DEBUG] Getting company identifier for {company_name}")
            company = COMPANIES[company_name]
            print(f"[DEBUG] Scraping news links for {company_name}")
            links = robust_scrape(company)
            print(f"[DEBUG] Scraped {len(links)} links for {company_name}")
            
            if not links:
                print(f"No links found for {company_name}")
                continue
            
            print(f"[DEBUG] Scraping articles for {company_name}")
            df = asyncio.get_event_loop().run_until_complete(scrape_all(links))
            print(f"[DEBUG] Scraped {len(df)} articles for {company_name}")
            
            if df.empty:
                print(f"No articles processed for {company_name}")
                continue
            
            print(f"[DEBUG] Sorting articles by date/time")
            df = df.sort_values(by=['publish_date', 'publish_time'], ascending=[False, False]).reset_index(drop=True)
            
            print(f"[DEBUG] Analyzing sentiment for {company_name}")
            predicted = analyze_sentiment(model, df)
            df["predicted"] = predicted
            print(f"[DEBUG] Sentiment analysis complete for {company_name}")
            
            df["sentiment"] = df["predicted"].apply(lambda x: x.split("\n")[1].strip() if len(x.split("\n")) > 1 else None)
            df["importance"] = df["predicted"].apply(lambda x: x.split("\n")[10].strip() if len(x.split("\n")) > 10 else None)
            df["summary"] = df["predicted"].apply(lambda x: x.split("\n")[4].strip() if len(x.split("\n")) > 4 else None)
            
            df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
            df = df[df['importance'].isin(['1', '2', '3', '4', '5'])]
            
            if df.empty:
                print(f"No valid sentiment analysis for {company_name}")
                continue
            
            print(f"[DEBUG] Translating summaries for {company_name}")
            translate = translate_summaries(model, df)
            df["translate"] = translate
            print(f"[DEBUG] Translation complete for {company_name}")
            
            if 'predicted' in df.columns:
                df.drop(columns=['predicted'], inplace=True)
            if 'body_text' in df.columns:
                df.drop(columns=['body_text'], inplace=True)
            
            now = datetime.now()
            date_time = now.strftime("%Y-%m-%d %H-%M").strip().replace(' ', '_')
            ticker = df['ticker'].iloc[0] if not df.empty else company_name
            filename = f"gemini_{ticker}_{date_time}.csv".lower()
            
            os.makedirs("../data/processed", exist_ok=True)
            df.to_csv(f"../data/processed/{filename}", index=False)
            print(f"[DEBUG] Saved CSV for {company_name}")
            
            complete_dict = df.to_dict(orient='records')
            result = collection.insert_many(complete_dict, ordered=True)
            print(f"[DEBUG] Inserted {len(result.inserted_ids)} documents for {company_name}")
            
            all_results.append(df)
            
        except Exception as e:
            print(f"Error processing {company_name}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print(f"Processing complete! Processed {len(all_results)} companies")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 