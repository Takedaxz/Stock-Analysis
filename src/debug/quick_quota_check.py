#!/usr/bin/env python3
"""
Quick Gemini API Quota Check
Simple script to check if API is available
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def quick_check():
    """Quick check if API quota is available"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return False
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content("hi")
        print("✅ API quota available")
        return True
    except Exception as e:
        error_str = str(e).lower()
        if "quota" in error_str or "429" in error_str:
            print("❌ Quota exceeded")
            return False
        else:
            print(f"⚠️  Other error: {e}")
            return True

if __name__ == "__main__":
    quick_check() 