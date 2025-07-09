# Stock Analysis Project - Reorganized Structure

## Overview
This is a comprehensive stock market analysis platform that combines data collection, sentiment analysis, technical indicators, quantitative modeling, and interactive visualizations for informed investment decisions.

## New Project Structure

```
Stock-Analysis/
├── src/                          # Source code
│   ├── data_collection/          # Data gathering modules
│   │   ├── yfinanceAPI.ipynb     # Yahoo Finance API integration
│   │   ├── ScrapingStockNews.ipynb
│   │   ├── ScrapingMarketNews.ipynb
│   │   ├── ScrapingTrendingNews.ipynb
│   │   ├── ScrapingFinancials.ipynb
│   │   ├── FinancialStatement.ipynb
│   │   ├── EarningsCalendar.ipynb
│   │   ├── EventCalendar.ipynb
│   │   ├── Index.ipynb
│   │   └── Data/                 # Collected data
│   │       ├── All_News/
│   │       ├── Earnings_Calendar/
│   │       ├── Event_Calendar/
│   │       ├── Financials/
│   │       ├── FinancialsAnalysis/
│   │       ├── Index/
│   │       ├── InfoFromYfinance/
│   │       ├── News/
│   │       └── Trending_News/
│   │
│   ├── sentiment_analysis/       # AI-powered sentiment analysis
│   │   ├── GeminiNewsAnalysis.ipynb
│   │   ├── DeepSeekNewsAnalysis.ipynb
│   │   ├── MistralNewsAnalysis.ipynb
│   │   ├── secret.env
│   │   └── Output/               # Sentiment analysis results
│   │
│   ├── technical_analysis/       # Technical indicators and analysis
│   │   ├── CandleStick.ipynb
│   │   ├── TradingDashboards.ipynb
│   │   ├── HypothesisTesting.ipynb
│   │   └── Indicators/           # Technical indicators
│   │       ├── main.py           # Main orchestrator
│   │       ├── SMA.ipynb
│   │       ├── EMA.ipynb
│   │       ├── RSI.ipynb
│   │       ├── MACD.ipynb
│   │       ├── BollingerBands.ipynb
│   │       ├── Stochastic_Oscillator.ipynb
│   │       ├── Momentum.ipynb
│   │       ├── Volume.ipynb
│   │       ├── SupportResistance.ipynb
│   │       ├── Ichimoku_Cloud.ipynb
│   │       ├── Elliott_Wave.ipynb
│   │       ├── Chart_Patterns.ipynb
│   │       └── Output/
│   │
│   ├── quantitative_analysis/    # ML models and statistical analysis
│   │   ├── VixSET50.ipynb
│   │   └── Basic/                # Basic quantitative models
│   │       ├── BasicPricePredicted.ipynb
│   │       ├── K-Means.ipynb
│   │       ├── YuantaML1.ipynb
│   │       ├── YusantaML2.ipynb
│   │       └── Data/
│   │
│   ├── visualization/            # Visualization components
│   │   ├── stock_app.py
│   │   ├── news_app.py
│   │   ├── financial_app.py
│   │   ├── EnglishToThai.ipynb
│   │   ├── backend/              # FastAPI backend
│   │   ├── src/                  # Frontend assets
│   │   └── ReturnCalculation/
│   │       ├── DCA.ipynb
│   │       └── Dividend.ipynb
│   │
│   └── utils/                    # Utility functions
│       ├── EDA.ipynb
│       ├── EnglishToThai.ipynb
│       ├── DCA.ipynb
│       └── Dividend.ipynb
│
├── apps/                         # Application deployments
│   ├── streamlit_apps/           # Streamlit applications
│   │   ├── stock_app.py
│   │   ├── news_app.py
│   │   └── financial_app.py
│   │
│   └── web_dashboard/            # Web dashboard
│       ├── backend/              # FastAPI backend
│       ├── src/                  # Frontend assets
│       └── requirements.txt
│
├── data/                         # Data storage
│   ├── raw/                      # Raw collected data
│   ├── processed/                # Processed data
│   │   ├── StockNews.ipynb
│   │   ├── TrendingNews.ipynb
│   │   └── Data/                # Processed data files
│   └── models/                   # Trained models
│
├── config/                       # Configuration files
│   └── requirements.txt
│
└── docs/                         # Documentation
    ├── README.md
    └── PROJECT_STRUCTURE.md
```

## Key Features by Module

### Data Collection (`src/data_collection/`)
- **Multi-source data gathering** from Yahoo Finance, news APIs, and financial websites
- **Automated scraping** of stock news, market news, and trending financial content
- **Financial statement analysis** with automated data extraction
- **Earnings and event calendar** tracking
- **Real-time market data** integration

### Sentiment Analysis (`src/sentiment_analysis/`)
- **Multi-model AI analysis** using Gemini, DeepSeek, and Mistral
- **News sentiment scoring** with importance weighting
- **Automated sentiment classification** (Positive/Negative/Neutral)
- **Thai language support** with translation capabilities

### Technical Analysis (`src/technical_analysis/`)
- **Comprehensive indicator suite** including RSI, MACD, Bollinger Bands, etc.
- **Advanced pattern recognition** for chart patterns and Elliott Waves
- **Support and resistance** level identification
- **Volume analysis** and momentum indicators
- **Interactive dashboards** for real-time analysis

### Quantitative Analysis (`src/quantitative_analysis/`)
- **Machine learning models** for price prediction
- **Clustering analysis** for market segmentation
- **Volatility analysis** using VIX and market indices
- **Statistical hypothesis testing** for trading strategies

### Visualization (`src/visualization/`)
- **Streamlit web applications** for interactive analysis
- **Real-time data visualization** with Plotly and Matplotlib
- **Candlestick charts** with technical indicators
- **Sentiment dashboards** with live updates
- **Return calculation tools** for investment analysis

### Applications (`apps/`)
- **Streamlit Apps** (`apps/streamlit_apps/`): Interactive web applications
- **Web Dashboard** (`apps/web_dashboard/`): FastAPI backend with frontend

## Usage Examples

### Running Streamlit Apps
```bash
cd apps/streamlit_apps
streamlit run stock_app.py
streamlit run news_app.py
streamlit run financial_app.py
```

### Running Technical Analysis
```bash
cd src/technical_analysis/Indicators
python main.py
```

### Running Sentiment Analysis
```bash
cd src/sentiment_analysis
jupyter notebook GeminiNewsAnalysis.ipynb
```

## Data Flow

1. **Data Collection** → Raw data stored in `data/raw/`
2. **Processing** → Processed data in `data/processed/`
3. **Analysis** → Results in respective module directories
4. **Visualization** → Interactive apps in `apps/`

## Configuration

### API Keys Required
- **OpenAI API** - For GPT-based sentiment analysis
- **Google Gemini API** - For Gemini sentiment analysis
- **Mistral AI API** - For Mistral sentiment analysis

### Environment Setup
Create a `.env` file or use the existing `secret.env` in the `src/sentiment_analysis/` directory. 