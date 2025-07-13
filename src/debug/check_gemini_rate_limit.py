#!/usr/bin/env python3
"""
Check Gemini API Rate Limits and Usage
Enhanced version with current status checking
"""

import os
import time
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import retry
import requests

# Load environment variables
load_dotenv()

def setup_gemini():
    """Setup Gemini API"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    generation_config = genai.GenerationConfig(temperature=0)
    return genai.GenerativeModel("gemini-2.5-flash-preview-04-17", generation_config=generation_config)

def check_current_usage():
    """Check current API usage and limits"""
    print("="*60)
    print("🔍 CHECKING CURRENT GEMINI API USAGE")
    print("="*60)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test API connectivity and get current status
    model = setup_gemini()
    
    # Test with minimal request
    test_prompt = "Hi"
    
    try:
        print("\n🔄 Testing API connectivity...")
        start_time = time.time()
        response = model.generate_content(test_prompt)
        end_time = time.time()
        
        print(f"✅ API is responsive!")
        print(f"   Response time: {end_time - start_time:.2f} seconds")
        print(f"   Response: '{response.text}'")
        
        return True
        
    except Exception as e:
        error_str = str(e).lower()
        print(f"❌ API Error: {e}")
        
        # Analyze the error type
        if "quota" in error_str:
            print("\n🚨 QUOTA EXCEEDED")
            print("   You've hit your daily request limit")
            print("   Free tier: 250 requests per day")
            print("   Quota resets daily at midnight UTC")
            return False
        elif "rate" in error_str:
            print("\n🚨 RATE LIMIT EXCEEDED")
            print("   You're making too many requests too quickly")
            print("   Free tier: 15 requests per minute")
            return False
        elif "429" in error_str:
            print("\n🚨 HTTP 429 - TOO MANY REQUESTS")
            print("   Rate limit exceeded")
            return False
        elif "403" in error_str:
            print("\n🚨 HTTP 403 - FORBIDDEN")
            print("   Possible quota exceeded or invalid API key")
            return False
        else:
            print(f"\n❓ UNKNOWN ERROR: {e}")
            return False

def test_rate_limits():
    """Test multiple requests to check rate limiting behavior"""
    print("\n" + "="*60)
    print("🧪 TESTING RATE LIMITS")
    print("="*60)
    
    model = setup_gemini()
    
    print("Testing multiple rapid requests...")
    successful_requests = 0
    failed_requests = 0
    
    for i in range(5):
        try:
            start_time = time.time()
            response = model.generate_content(f"Test request {i+1}")
            end_time = time.time()
            
            successful_requests += 1
            print(f"   Request {i+1}: ✅ Success ({end_time - start_time:.2f}s)")
            
            # Small delay between requests
            time.sleep(0.5)
            
        except Exception as e:
            failed_requests += 1
            error_str = str(e).lower()
            print(f"   Request {i+1}: ❌ Failed - {e}")
            
            if "quota" in error_str:
                print("      → Daily quota exceeded")
                break
            elif "rate" in error_str:
                print("      → Rate limit hit")
                break
    
    print(f"\n📊 Test Results:")
    print(f"   Successful requests: {successful_requests}")
    print(f"   Failed requests: {failed_requests}")
    
    if failed_requests == 0:
        print("   ✅ Rate limits look good!")
        return True
    else:
        print("   ⚠️  Rate limits detected")
        return False

def calculate_news_processor_usage():
    """Calculate how many API calls the news processor will make"""
    print("\n" + "="*60)
    print("📊 NEWS PROCESSOR API USAGE ESTIMATE")
    print("="*60)
    
    # Current configuration
    MAX_ARTICLES = 20
    FINAL_ARTICLES = 10
    
    # API calls per article
    sentiment_analysis_calls = FINAL_ARTICLES  # 1 call per article
    translation_calls = FINAL_ARTICLES  # 1 call per article
    
    total_calls = sentiment_analysis_calls + translation_calls
    
    print(f"📈 Estimated API calls per run:")
    print(f"   Articles to process: {FINAL_ARTICLES}")
    print(f"   Sentiment analysis: {sentiment_analysis_calls} calls")
    print(f"   Translation: {translation_calls} calls")
    print(f"   Total calls per run: {total_calls}")
    
    # Daily limits
    daily_limit = 250
    remaining_calls = daily_limit - total_calls
    
    print(f"\n📅 Daily quota analysis:")
    print(f"   Daily limit: {daily_limit} calls")
    print(f"   Calls per run: {total_calls}")
    print(f"   Remaining after run: {remaining_calls}")
    
    if remaining_calls >= 0:
        print(f"   ✅ Safe to run! ({remaining_calls} calls remaining)")
        
        # Calculate how many runs per day
        runs_per_day = daily_limit // total_calls
        print(f"   📊 Maximum runs per day: {runs_per_day}")
        
    else:
        print(f"   ❌ Would exceed daily limit!")
        print(f"   💡 Consider reducing FINAL_ARTICLES to {daily_limit // 2}")
    
    return remaining_calls >= 0

def show_recommendations():
    """Show recommendations based on current status"""
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    
    print("1. 🕐 If quota exceeded:")
    print("   - Wait for daily reset (midnight UTC)")
    print("   - Check usage at: https://aistudio.google.com/app/apikey")
    print("   - Consider upgrading to Pro for higher limits")
    
    print("\n2. ⚡ For rate limiting:")
    print("   - Add delays between API calls (2-3 seconds)")
    print("   - Use exponential backoff for retries")
    print("   - Reduce batch sizes")
    
    print("\n3. 🔧 For your news processor:")
    print("   - Add time.sleep(2) between requests")
    print("   - Reduce FINAL_ARTICLES to 5-10")
    print("   - Add graceful error handling for quota limits")
    print("   - Consider using gemini-1.5-flash (different limits)")
    
    print("\n4. 📊 Monitor usage:")
    print("   - Run this script before each workflow")
    print("   - Check Google AI Studio dashboard regularly")
    print("   - Set up alerts for quota warnings")

def main():
    print("🚀 GEMINI API RATE LIMIT CHECKER")
    print("="*60)
    
    # Check current usage
    api_working = check_current_usage()
    
    if api_working:
        # Test rate limits
        rate_limits_ok = test_rate_limits()
        
        # Calculate news processor usage
        safe_to_run = calculate_news_processor_usage()
        
        print("\n" + "="*60)
        print("🎯 SUMMARY")
        print("="*60)
        
        if safe_to_run and rate_limits_ok:
            print("✅ READY TO RUN NEWS PROCESSOR")
            print("   - API is working")
            print("   - Rate limits are acceptable")
            print("   - Quota allows for processing")
        elif safe_to_run:
            print("⚠️  CAUTION - RATE LIMITS DETECTED")
            print("   - API is working")
            print("   - Add delays between requests")
            print("   - Monitor for rate limit errors")
        else:
            print("❌ QUOTA EXCEEDED")
            print("   - Wait for daily reset")
            print("   - Reduce article count")
            print("   - Consider upgrading to Pro")
    else:
        print("\n❌ API NOT AVAILABLE")
        print("   - Check your API key")
        print("   - Wait for quota reset")
        print("   - Verify account status")
    
    show_recommendations()
    
    print("\n" + "="*60)
    print("🔗 Useful Links:")
    print("   - Google AI Studio: https://aistudio.google.com/app/apikey")
    print("   - Rate Limits: https://ai.google.dev/gemini-api/docs/rate-limits")
    print("   - Pricing: https://ai.google.dev/gemini-api/docs/pricing")

if __name__ == "__main__":
    main() 