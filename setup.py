#!/usr/bin/env python3
"""
Stock Analysis Platform Setup Script
This script helps set up the project structure and dependencies.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "src/data_collection/Data",
        "src/sentiment_analysis/Output",
        "src/technical_analysis/Indicators/Output",
        "src/quantitative_analysis/Basic/Data",
        "data/raw",
        "data/processed",
        "data/models",
        "apps/streamlit_apps",
        "apps/web_dashboard",
        "config",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_requirements():
    """Install required packages."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "config/requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️  Some dependencies may not have installed correctly. Please check manually.")

def create_env_template():
    """Create environment template file."""
    env_template = """# API Keys for Stock Analysis Platform
# Copy this file to secret.env and fill in your API keys

# OpenAI API (for GPT-based sentiment analysis)
OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini API (for Gemini sentiment analysis)
GOOGLE_API_KEY=your_google_api_key_here

# Mistral AI API (for Mistral sentiment analysis)
MISTRAL_API_KEY=your_mistral_api_key_here

# MongoDB Connection String (for web dashboard)
MONGO_CONNECTION_STRING=mongodb://localhost:27017

# Optional: Additional API keys
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
# FINNHUB_API_KEY=your_finnhub_key_here
"""
    
    env_path = "src/sentiment_analysis/secret.env.template"
    with open(env_path, "w") as f:
        f.write(env_template)
    print(f"✓ Created environment template: {env_path}")

def create_run_scripts():
    """Create convenient run scripts."""
    
    # Streamlit apps runner
    streamlit_runner = """#!/bin/bash
# Streamlit Apps Runner
echo "Starting Stock Analysis Streamlit Apps..."

echo "1. Stock Analysis App"
cd apps/streamlit_apps
streamlit run stock_app.py &

echo "2. News Sentiment App"
streamlit run news_app.py &

echo "3. Financial Analysis App"
streamlit run financial_app.py &

echo "All apps started! Check your browser for the applications."
"""
    
    with open("run_streamlit_apps.sh", "w") as f:
        f.write(streamlit_runner)
    os.chmod("run_streamlit_apps.sh", 0o755)
    print("✓ Created run script: run_streamlit_apps.sh")
    
    # Technical analysis runner
    ta_runner = """#!/bin/bash
# Technical Analysis Runner
echo "Running Technical Analysis..."

cd src/technical_analysis/Indicators
python main.py

echo "Technical analysis completed!"
"""
    
    with open("run_technical_analysis.sh", "w") as f:
        f.write(ta_runner)
    os.chmod("run_technical_analysis.sh", 0o755)
    print("✓ Created run script: run_technical_analysis.sh")

def main():
    """Main setup function."""
    print("🚀 Setting up Stock Analysis Platform...\n")
    
    # Create directories
    print("📁 Creating project directories...")
    create_directories()
    
    # Install requirements
    print("\n📦 Installing dependencies...")
    install_requirements()
    
    # Create environment template
    print("\n🔑 Creating environment template...")
    create_env_template()
    
    # Create run scripts
    print("\n📜 Creating run scripts...")
    create_run_scripts()
    
    print("\n✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Copy src/sentiment_analysis/secret.env.template to src/sentiment_analysis/secret.env")
    print("2. Edit secret.env with your API keys")
    print("3. Run ./run_streamlit_apps.sh to start the applications")
    print("4. Run ./run_technical_analysis.sh to run technical analysis")
    print("\n📚 Check docs/PROJECT_STRUCTURE.md for detailed documentation")

if __name__ == "__main__":
    main() 