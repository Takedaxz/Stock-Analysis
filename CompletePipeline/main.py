import os
import re
import time
import random
import asyncio
import warnings
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import nest_asyncio
import cloudscraper
from htmldate import find_date
from bs4 import BeautifulSoup
from newspaper import Article
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import google.generativeai as genai
from tqdm import tqdm
from google.api_core import retry
import numpy as np
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

MAX_WORKERS = 10
MAX_RETRIES = 8
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "ngrok-skip-browser-warning":"1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
    "Referer": "https://www.investing.com/",
    "DNT": "1"
}

scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'mobile': False
    },
    delay=2,
)

FETCH_WORKERS = min(32, os.cpu_count() * 4)
PROCESS_WORKERS = os.cpu_count() or 4
MAX_FETCH_RETRIES = 5
RETRY_DELAY = 1

def fetch_page(page: int):
    url = "https://www.investing.com/news/most-popular-news"
    if page > 1:
        url += f"/{page}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = scraper.get(url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
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
                logging.error(f"Page {page} failed after {MAX_RETRIES}: {e}")
    return []

def robust_scrape(max_pages=10):
    first = fetch_page(1)
    PER_PAGE = len(first)
    if PER_PAGE == 0:
        raise RuntimeError("Failed to fetch the first page. Please check headers or cookies.")
    logging.info(f"Fetched {PER_PAGE} links on page 1")
    results = {1: first}
    if max_pages == 1 or PER_PAGE == 0:
        logging.info("Only one page detected or no pagination found.")
        return first
    pages = list(range(2, max_pages + 1))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_page, p): p for p in pages}
        for fut in as_completed(futures):
            p = futures[fut]
            results[p] = fut.result()
        for round in range(1, MAX_RETRIES + 1):
            bad = [p for p, links in results.items() if len(links) != PER_PAGE]
            if not bad:
                logging.info(f"All pages OK after {round - 1} retries")
                break
            logging.info(f"Retry round {round} for pages: {bad}")
            futures = {pool.submit(fetch_page, p): p for p in bad}
            for fut in as_completed(futures):
                p = futures[fut]
                results[p] = fut.result()
        else:
            logging.warning("Retry limit reached; some pages may still be incomplete.")
    total_fetched = sum(len(links) for links in results.values())
    expected = PER_PAGE * max_pages
    logging.info(f"Total links fetched (including duplicates): {total_fetched} (expected {expected})")
    all_links = set(link for links in results.values() for link in links)
    logging.info(f"Final: got {len(all_links)} unique URLs")
    return list(all_links)

def is_placeholder(html: str) -> bool:
    lower = html.lower() if html else ""
    return (
        'temporarily down for maintenance' in lower
        or 'just a moment' in lower
        or "we're temporarily down" in lower
    )

def safe_find_datetime(url, html_content=None):
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
    for attempt in range(1, MAX_FETCH_RETRIES + 1):
        try:
            resp = scraper.get(url, timeout=30)
            html = resp.text
            if is_placeholder(html):
                raise RuntimeError('Placeholder')
            logging.info(f"[Fetch][{idx}/{total}][ok]")
            return url, html
        except Exception:
            logging.warning(f"[Fetch][{idx}/{total}][retry {attempt}]")
            if attempt < MAX_FETCH_RETRIES:
                time.sleep(RETRY_DELAY)
    logging.error(f"[Fetch error] {idx}/{total}: failed after {MAX_FETCH_RETRIES} retries")
    return url, None

def process_article(arg):
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
    return {'ticker':'news','publish_date': date, 'publish_time': tm,
             'title': title, 'body_text': text, 'url': url}

async def scrape_all(urls):
    total = len(urls)
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as fetch_pool:
        fetch_tasks = [loop.run_in_executor(fetch_pool, fetch_html, u, i+1, total)
                       for i, u in enumerate(urls)]
        fetched = await asyncio.gather(*fetch_tasks)
    records = []
    with ThreadPoolExecutor(max_workers=PROCESS_WORKERS) as proc_pool:
        futures = {
            proc_pool.submit(process_article, fr): fr[0]
            for fr in fetched if fr[1]
        }
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            logging.info(f"[Process][{i}/{total}] {futures[fut]}")
            if res:
                records.append(res)
    return pd.DataFrame(records)

def is_retryable(e) -> bool:
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
    return model.generate_content(prompt).text

def main():
    # Read config from environment variables
    max_pages = int(os.environ.get('MAX_PAGES', 2))
    env_path = os.environ.get('ENV_PATH', '../SentimentAnalysis/GPT/secret.env')
    nest_asyncio.apply()
    # Step 1: Scrape links
    links = robust_scrape(max_pages=max_pages)
    links = [f'https://www.investing.com{l}' if l.startswith('/') else l for l in links]
    # Step 2: Scrape article content
    df = asyncio.get_event_loop().run_until_complete(scrape_all(links))
    df = df.sort_values(by=['publish_date', 'publish_time'], ascending=[False,False]).reset_index(drop=True)
    pd.set_option('display.max_columns', None)
    empty_body_count = df[df['body_text'] == ''].shape[0]
    logging.info(f"Number of articles with empty body_text: {empty_body_count}")
    # Step 3: Load API keys and MongoDB connection
    load_dotenv(env_path)
    api_key = os.getenv("GEMINI_API_KEY")
    mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")
    if api_key is None:
        logging.error("Error: GEMINI_API_KEY not found in .env file or environment variables.")
        return (json.dumps({"status": "error", "message": "GEMINI_API_KEY not found"}), 500, {'Content-Type': 'application/json'})
    if mongo_connection_string is None:
        logging.error("Error: MONGO_CONNECTION_STRING not found in .env file or environment variables.")
        return (json.dumps({"status": "error", "message": "MONGO_CONNECTION_STRING not found"}), 500, {'Content-Type': 'application/json'})
    try:
        client = MongoClient(mongo_connection_string)
        db = client['stock_news_db']
        collection = db['news_data']
        client.admin.command('ping')
        logging.info("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return (json.dumps({"status": "error", "message": str(e)}), 500, {'Content-Type': 'application/json'})
    # Step 4: LLM Analysis
    genai.configure(api_key=api_key)
    generation_config = genai.GenerationConfig(temperature=0)
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17", generation_config=generation_config)
    prompt_template = """You are a financial news analyst specializing in stock market impact. Your task is to analyze the provided news article, summarize its core content concisely, determine its sentiment (positive, negative, or neutral), and assess its importance to the specified stock.\n\nHere is the news from stock [STOCK] title and body:\n---\n[TITLE]\n---\n[BODY]\n---\n\nPlease provide your analysis in the following format (Don't forget to make space between the sections as shown):\n\n**Sentiment:**\n[Positive / Negative / Neutral]\n\n**Summary:**\n[Your concise summary of the article, typically 2-3 sentences.]\n\n**Reasoning for Sentiment:**\n[Brief explanation (1-2 sentences) of why you categorized the sentiment as such, referencing key points or tone from the article.]\n\n**Importance to Stock [STOCK]:**\n[1-5, where 1 is minimal importance and 5 is very high importance.Answer in 1-5 only, no explanation.] (Answer only in number 1-5)\n\n**Reasoning for Importance:**\n[Brief explanation (1-2 sentences) of why you assigned this importance score, referencing specific details from the article that would impact the stock.]"""
    predicted = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Prompting"):
        current_stock = row.get("ticker", "news")
        filled_prompt = prompt_template.replace("[STOCK]", current_stock)
        filled_prompt = filled_prompt.replace("[TITLE]", row["title"])
        filled_prompt = filled_prompt.replace("[BODY]", row["body_text"])
        try:
            response = generate_content_with_rate_limit(model, filled_prompt)
            finalprediction = response.strip()
            if not finalprediction:
                logging.warning(f"Row {index}: LLM returned an empty string.")
                predicted.append("LLM_EMPTY_RESPONSE")
            else:
                predicted.append(finalprediction)
        except ValueError as ve:
            logging.error(f"Row {index}: ValueError - {ve}. Appending 'ERROR_VALUE_ERROR'.")
            predicted.append("ERROR_VALUE_ERROR")
            continue
        except Exception as e:
            if "429 Too Many Requests" in str(e) or "quota" in str(e).lower():
                logging.error(f"Row {index}: Rate Limit Exceeded or Quota Error - {e}. Appending 'ERROR_RATE_LIMIT'.")
                predicted.append("ERROR_RATE_LIMIT")
            elif "safety" in str(e).lower() or "blocked" in str(e).lower():
                 logging.error(f"Row {index}: Content Safety/Blocked - {e}. Appending 'ERROR_SAFETY_BLOCKED'.")
                 predicted.append("ERROR_SAFETY_BLOCKED")
            else:
                logging.error(f"Row {index}: Unexpected Error - {e}. Appending 'ERROR_UNEXPECTED'.")
                predicted.append("ERROR_UNEXPECTED")
            continue
    predicted = np.array(predicted)
    df["predicted"] = predicted
    df["sentiment"] = df["predicted"].apply(lambda x: x.split("\n")[1].strip() if len(x.split("\n")) > 1 else None)
    df["importance"] = df["predicted"].apply(lambda x: x.split("\n")[10].strip() if len(x.split("\n")) > 10 else None)
    df["summary"] = df["predicted"].apply(lambda x: x.split("\n")[4].strip() if len(x.split("\n")) > 4 else None)
    df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
    df = df[df['importance'].isin(['1', '2', '3', '4', '5'])]
    # Step 5: Translate summary
    model_translate = genai.GenerativeModel("gemini-2.0-flash-001", generation_config=generation_config)
    prompt_translate = """Translate the following English sentence to Thai. Do not translate proper nouns, company names, product names, abbreviations, or technical terms — keep them in English. Do not provide any explanation, just the translation.\n[TEXT]"""
    translate = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Prompting-Translate"):
        filled_prompt = prompt_translate.replace("[TEXT]", row["summary"])
        try:
            response = generate_content_with_rate_limit(model_translate, filled_prompt)
            finalprediction = response.strip()
            if not finalprediction:
                logging.warning(f"Row {index}: LLM returned an empty string.")
                translate.append("LLM_EMPTY_RESPONSE")
            else:
                translate.append(finalprediction)
        except ValueError as ve:
            logging.error(f"Row {index}: ValueError - {ve}. Appending 'ERROR_VALUE_ERROR'.")
            translate.append("ERROR_VALUE_ERROR")
            continue
        except Exception as e:
            if "429 Too Many Requests" in str(e) or "quota" in str(e).lower():
                logging.error(f"Row {index}: Rate Limit Exceeded or Quota Error - {e}. Appending 'ERROR_RATE_LIMIT'.")
                translate.append("ERROR_RATE_LIMIT")
            elif "safety" in str(e).lower() or "blocked" in str(e).lower():
                 logging.error(f"Row {index}: Content Safety/Blocked - {e}. Appending 'ERROR_SAFETY_BLOCKED'.")
                 translate.append("ERROR_SAFETY_BLOCKED")
            else:
                logging.error(f"Row {index}: Unexpected Error - {e}. Appending 'ERROR_UNEXPECTED'.")
                translate.append("ERROR_UNEXPECTED")
            continue
    df["translate"] = translate
    # Only insert to MongoDB, no CSV
    complete_dict = df.to_dict(orient='records')
    try:
        result = collection.insert_many(complete_dict, ordered=True)
        logging.info(f"Successfully inserted document with id: {result.inserted_ids}")
        return (json.dumps({"status": "success", "inserted_count": len(result.inserted_ids)}), 200, {'Content-Type': 'application/json'})
    except Exception as e:
        logging.error(f"MongoDB insert error: {e}")
        return (json.dumps({"status": "error", "message": str(e)}), 500, {'Content-Type': 'application/json'})
    
if __name__ == "__main__":
    main()