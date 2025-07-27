#!/usr/bin/env python3
"""
MongoDB Duplicate Removal Script
Removes duplicate topics from the stock_news_db database using advanced duplicate detection
"""

import os
import re
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import warnings

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Duplicate detection settings - EXACT MATCHES ONLY
ENABLE_DETAILED_LOGGING = True

def setup_mongodb():
    """Setup MongoDB connection"""
    print("Setting up MongoDB connection...")
    mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")
    if not mongo_connection_string:
        raise ValueError("MONGO_CONNECTION_STRING not found in environment variables")
    
    client = MongoClient(mongo_connection_string)
    db = client['stock_news_db']
    collection = db['news_data']
    
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
    return collection



def is_duplicate_article(article1, article2):
    """Check if two articles are duplicates based on exact URL and title matches only"""
    # Check exact URL match first (fast check)
    url1 = article1.get('url', '').lower()
    url2 = article2.get('url', '').lower()
    
    # If URLs are identical, it's a duplicate
    if url1 and url2 and url1 == url2:
        return True
    
    # Check exact title match
    title1 = article1.get('title', '').strip()
    title2 = article2.get('title', '').strip()
    
    if title1 and title2 and title1.lower() == title2.lower():
        return True
    
    return False

def find_duplicates_in_mongodb(collection):
    """Find and return duplicate articles from MongoDB"""
    print("Fetching all articles from MongoDB...")
    
    # Get all articles from MongoDB
    all_articles = list(collection.find({}))
    print(f"Total articles in database: {len(all_articles)}")
    
    if not all_articles:
        print("No articles found in database")
        return []
    
    # Group articles by ticker for more efficient processing
    articles_by_ticker = {}
    for article in all_articles:
        ticker = article.get('ticker', 'unknown')
        if ticker not in articles_by_ticker:
            articles_by_ticker[ticker] = []
        articles_by_ticker[ticker].append(article)
    
    print(f"Articles grouped by {len(articles_by_ticker)} tickers")
    
    duplicates_to_remove = []
    processed_count = 0
    
    for ticker, articles in articles_by_ticker.items():
        print(f"\nProcessing {ticker}: {len(articles)} articles")
        
        # Sort articles by date (newest first) to keep the most recent
        articles.sort(key=lambda x: x.get('publish_date', ''), reverse=True)
        
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            url = article.get('url', '').lower()
            title = article.get('title', '')
            
            # Skip if URL is already seen
            if url in seen_urls:
                duplicates_to_remove.append(article['_id'])
                if ENABLE_DETAILED_LOGGING:
                    print(f"  Duplicate found (URL): {title[:50]}...")
                continue
            
            # Check if this article is a duplicate of any existing unique article
            is_duplicate = False
            duplicate_reason = ""
            
            for unique_article in unique_articles:
                if is_duplicate_article(article, unique_article):
                    is_duplicate = True
                    # Determine the reason for duplication
                    url1 = article.get('url', '').lower()
                    url2 = unique_article.get('url', '').lower()
                    title1 = article.get('title', '').strip()
                    title2 = unique_article.get('title', '').strip()
                    
                    if url1 and url2 and url1 == url2:
                        duplicate_reason = "exact URL"
                    elif title1 and title2 and title1.lower() == title2.lower():
                        duplicate_reason = "exact title"
                    else:
                        duplicate_reason = "exact match"
                    
                    break
            
            if is_duplicate:
                duplicates_to_remove.append(article['_id'])
                if ENABLE_DETAILED_LOGGING:
                    print(f"  Duplicate found ({duplicate_reason}): {title[:50]}...")
            else:
                unique_articles.append(article)
                if url:
                    seen_urls.add(url)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count} articles...")
    
    print(f"\nFound {len(duplicates_to_remove)} duplicate articles to remove")
    return duplicates_to_remove

def remove_duplicates_from_mongodb(collection, duplicate_ids):
    """Remove duplicate articles from MongoDB"""
    if not duplicate_ids:
        print("No duplicates to remove")
        return 0
    
    print(f"Removing {len(duplicate_ids)} duplicate articles from MongoDB...")
    
    # Remove duplicates in batches to avoid memory issues
    batch_size = 100
    removed_count = 0
    
    for i in range(0, len(duplicate_ids), batch_size):
        batch = duplicate_ids[i:i + batch_size]
        result = collection.delete_many({"_id": {"$in": batch}})
        removed_count += result.deleted_count
        print(f"  Removed batch {i//batch_size + 1}: {result.deleted_count} articles")
    
    print(f"Successfully removed {removed_count} duplicate articles")
    return removed_count

def backup_collection(collection):
    """Create a backup of the collection before removing duplicates"""
    print("Creating backup of collection...")
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_collection_name = f"news_data_backup_{timestamp}"
    
    # Create backup collection
    db = collection.database
    backup_collection = db[backup_collection_name]
    
    # Copy all documents to backup collection
    all_docs = list(collection.find({}))
    if all_docs:
        backup_collection.insert_many(all_docs)
        print(f"Backup created: {backup_collection_name} with {len(all_docs)} documents")
    else:
        print("No documents to backup")
    
    return backup_collection_name

def get_collection_stats(collection):
    """Get statistics about the collection"""
    print("\n" + "="*50)
    print("COLLECTION STATISTICS")
    print("="*50)
    
    total_docs = collection.count_documents({})
    print(f"Total documents: {total_docs}")
    
    # Count by ticker
    pipeline = [
        {"$group": {"_id": "$ticker", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    
    ticker_counts = list(collection.aggregate(pipeline))
    print(f"\nDocuments by ticker:")
    for ticker_count in ticker_counts:
        ticker = ticker_count['_id'] or 'unknown'
        count = ticker_count['count']
        print(f"  {ticker}: {count}")
    
    # Count by sentiment
    sentiment_pipeline = [
        {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    
    sentiment_counts = list(collection.aggregate(sentiment_pipeline))
    print(f"\nDocuments by sentiment:")
    for sentiment_count in sentiment_counts:
        sentiment = sentiment_count['_id'] or 'unknown'
        count = sentiment_count['count']
        print(f"  {sentiment}: {count}")

def main():
    print("="*60)
    print("MONGODB DUPLICATE REMOVAL TOOL")
    print("="*60)
    
    try:
        # Setup MongoDB connection
        collection = setup_mongodb()
        
        # Show current statistics
        get_collection_stats(collection)
        
        # Ask user for confirmation
        print("\n" + "="*50)
        print("DUPLICATE REMOVAL OPTIONS")
        print("="*50)
        print("1. Create backup and remove duplicates")
        print("2. Remove duplicates without backup (NOT RECOMMENDED)")
        print("3. Only find duplicates (dry run)")
        print("4. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                # Create backup and remove duplicates
                backup_name = backup_collection(collection)
                print(f"\nBackup created: {backup_name}")
                
                # Find and remove duplicates
                duplicate_ids = find_duplicates_in_mongodb(collection)
                if duplicate_ids:
                    removed_count = remove_duplicates_from_mongodb(collection, duplicate_ids)
                    print(f"\n✅ Successfully removed {removed_count} duplicate articles")
                    print(f"📦 Backup available in collection: {backup_name}")
                else:
                    print("\n✅ No duplicates found")
                break
                
            elif choice == "2":
                # Remove duplicates without backup
                confirm = input("⚠️  WARNING: This will permanently delete duplicates without backup. Continue? (yes/no): ")
                if confirm.lower() == 'yes':
                    duplicate_ids = find_duplicates_in_mongodb(collection)
                    if duplicate_ids:
                        removed_count = remove_duplicates_from_mongodb(collection, duplicate_ids)
                        print(f"\n✅ Successfully removed {removed_count} duplicate articles")
                    else:
                        print("\n✅ No duplicates found")
                else:
                    print("Operation cancelled")
                break
                
            elif choice == "3":
                # Dry run - only find duplicates
                print("\n🔍 DRY RUN - Finding duplicates only...")
                duplicate_ids = find_duplicates_in_mongodb(collection)
                if duplicate_ids:
                    print(f"\n📊 Found {len(duplicate_ids)} duplicate articles")
                    print("Run option 1 to create backup and remove them")
                else:
                    print("\n✅ No duplicates found")
                break
                
            elif choice == "4":
                print("Exiting...")
                break
                
            else:
                print("Please enter a valid choice (1-4)")
        
        # Show final statistics
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        get_collection_stats(collection)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your MongoDB connection and try again")

if __name__ == "__main__":
    main() 