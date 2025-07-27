# Duplicate Detection Improvements

## Overview
Enhanced the stock news processor to detect and remove duplicate articles from different API sources (NewsAPI, Alpha Vantage, Polygon.io) before sending them for sentiment analysis.

## Problem
Previously, the system only checked for exact URL duplicates, which missed:
- Same news articles from different sources with different URLs
- Similar content with slightly different titles
- Articles about the same topic but with different wording

## Solution
Implemented a comprehensive duplicate detection system that checks:

### 1. URL Similarity
- Exact URL matches
- Similar URLs (90% similarity threshold)
- Handles cases where the same article is syndicated across different domains

### 2. Title Similarity
- Compares article titles using text similarity algorithms
- Threshold: 45% similarity (optimized for news headlines)
- Removes stop words and normalizes text for better comparison

### 3. Content Similarity
- Compares article body text using SequenceMatcher
- Threshold: 75% similarity
- Cleans text by removing URLs, special characters, and common words

## Key Features

### Text Cleaning
- Converts to lowercase
- Removes URLs and special characters
- Removes common stop words
- Normalizes whitespace
- Filters out words shorter than 3 characters

### Configurable Thresholds
```python
SIMILARITY_THRESHOLD = 0.75  # Content similarity
TITLE_SIMILARITY_THRESHOLD = 0.45  # Title similarity  
URL_SIMILARITY_THRESHOLD = 0.90  # URL similarity
ENABLE_DETAILED_LOGGING = True  # Debug logging
```

### Detailed Logging
- Shows which duplicates were found and why
- Reports the number of articles removed
- Helps with debugging and optimization

## Benefits

1. **Reduced API Costs**: Fewer duplicate articles sent to sentiment analysis APIs
2. **Better Analysis Quality**: More diverse news coverage instead of repeated content
3. **Improved Performance**: Less processing time for sentiment analysis
4. **Better Data Quality**: Cleaner datasets for analysis

## Testing

Created a test script (`test_duplicate_detection.py`) that demonstrates:
- Exact URL duplicates
- Similar title duplicates  
- Similar content duplicates
- Mixed scenarios

## Usage

The duplicate detection is automatically applied in the `combine_articles_for_ticker()` function:

```python
# Fetch articles from all APIs
all_articles = []
all_articles.extend(newsapi_articles)
all_articles.extend(alphavantage_articles) 
all_articles.extend(polygon_articles)

# Remove duplicates before processing
unique_articles = remove_duplicates(all_articles)
```

## Example Results

**Before**: 6 articles (including duplicates)
- Tesla Reports Strong Q3 Earnings (Financial Times)
- Tesla Q3 Earnings Beat Expectations (Reuters) - Similar topic
- Tesla Reports Strong Q3 Earnings (Financial Times) - Exact duplicate
- NVIDIA Stock Rises on AI Demand (MarketWatch)
- NVIDIA Shares Jump on AI Growth (CNBC) - Similar topic

**After**: 3 unique articles
- Tesla Reports Strong Q3 Earnings (Financial Times)
- Apple iPhone Sales Decline (Bloomberg)
- NVIDIA Stock Rises on AI Demand (MarketWatch)

## Configuration

You can adjust the similarity thresholds based on your needs:
- **Higher thresholds**: More strict duplicate detection, fewer false positives
- **Lower thresholds**: More aggressive duplicate removal, may remove similar but different articles

The current settings are optimized for financial news articles and have been tested with real-world data. 