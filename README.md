# Stock Analysis Platform

A comprehensive stock market analysis platform that combines data collection, sentiment analysis, technical indicators, quantitative modeling, and interactive visualizations for informed investment decisions.

## 🚀 Live Applications

- **News Sentiment Dashboard**: https://stock-analysis-theta-inky.vercel.app

## 📁 Project Structure

```
Stock-Analysis/
├── src/                          # Source code
│   ├── data_collection/          # Data gathering modules
│   │   ├── yahoo_finance_api.ipynb
│   │   ├── stock_news_scraper.ipynb
│   │   ├── market_news_scraper.ipynb
│   │   ├── trending_news_scraper.ipynb
│   │   ├── financial_data_scraper.ipynb
│   │   ├── financial_statement_analyzer.ipynb
│   │   ├── earnings_calendar_scraper.ipynb
│   │   ├── event_calendar_scraper.ipynb
│   │   ├── market_index_scraper.ipynb
│   │   ├── stock_news_processor.ipynb
│   │   ├── trending_news_processor.ipynb
│   │   └── data/                 # Collected data
│   │       ├── all_news/
│   │       ├── earnings_calendar/
│   │       ├── event_calendar/
│   │       ├── financials/
│   │       ├── financials_analysis/
│   │       ├── stock_index/
│   │       ├── info_from_yfinance/
│   │       ├── stock_news/
│   │       └── trending_news/
│   │
│   ├── sentiment_analysis/       # AI-powered sentiment analysis
│   │   ├── gemini_sentiment_analyzer.ipynb
│   │   ├── deepseek_sentiment_analyzer.ipynb
│   │   ├── mistral_sentiment_analyzer.ipynb
│   │   ├── secret.env
│   │   └── output/               # Sentiment analysis results
│   │       ├── deepseek/
│   │       ├── gemini/
│   │       └── mistral/
│   │
│   ├── technical_analysis/       # Technical indicators and analysis
│   │   ├── candlestick_chart_analyzer.ipynb
│   │   ├── trading_dashboard.ipynb
│   │   ├── statistical_hypothesis_testing.ipynb
│   │   └── indicators/           # Technical indicators
│   │       ├── main.py           # Main orchestrator
│   │       ├── sma.ipynb
│   │       ├── ema.ipynb
│   │       ├── rsi.ipynb
│   │       ├── macd.ipynb
│   │       ├── bollinger_bands.ipynb
│   │       ├── stochastic_oscillator.ipynb
│   │       ├── momentum.ipynb
│   │       ├── volume.ipynb
│   │       ├── support_resistance.ipynb
│   │       ├── ichimoku_cloud.ipynb
│   │       ├── elliott_wave.ipynb
│   │       ├── chart_patterns.ipynb
│   │       └── output/
│   │
│   ├── quantitative_analysis/    # ML models and statistical analysis
│   │   ├── vix_set50.ipynb
│   │   └── basic/                # Basic quantitative models
│   │       ├── basic_price_prediction.ipynb
│   │       ├── k_means.ipynb
│   │       ├── yuanta_ml1.ipynb
│   │       ├── yuanta_ml2.ipynb
│   │       └── data/
│   │
│   ├── visualization/            # Visualization components
│   │   ├── backend/              # FastAPI backend
│   │   ├── src/                  # Frontend assets
│   │   └── return_calculation/
│   │       ├── dca.ipynb
│   │       └── dividend.ipynb
│   │
│   └── utils/                    # Utility functions
│       ├── eda.ipynb
│       ├── english_to_thai.ipynb
│       ├── dca.ipynb
│       └── dividend.ipynb
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
    └── project_structure.md
```

## 🎯 Key Features

### 📊 Data Collection (`src/data_collection/`)
- **Multi-source data gathering** from Yahoo Finance, news APIs, and financial websites
- **Automated scraping** of stock news, market news, and trending financial content
- **Financial statement analysis** with automated data extraction
- **Earnings and event calendar** tracking
- **Real-time market data** integration

### 🤖 Sentiment Analysis (`src/sentiment_analysis/`)
- **Multi-model AI analysis** using Gemini, DeepSeek, and Mistral
- **News sentiment scoring** with importance weighting
- **Automated sentiment classification** (Positive/Negative/Neutral)
- **Thai language support** with translation capabilities

### 📈 Technical Analysis (`src/technical_analysis/`)
- **Comprehensive indicator suite** including RSI, MACD, Bollinger Bands, etc.
- **Advanced pattern recognition** for chart patterns and Elliott Waves
- **Support and resistance** level identification
- **Volume analysis** and momentum indicators
- **Interactive dashboards** for real-time analysis

### 🧮 Quantitative Analysis (`src/quantitative_analysis/`)
- **Machine learning models** for price prediction
- **Clustering analysis** for market segmentation
- **Volatility analysis** using VIX and market indices
- **Statistical hypothesis testing** for trading strategies

### 🎨 Visualization (`src/visualization/`)
- **Streamlit web applications** for interactive analysis
- **Real-time data visualization** with Plotly and Matplotlib
- **Candlestick charts** with technical indicators
- **Sentiment dashboards** with live updates
- **Return calculation tools** for investment analysis

### 📱 Applications (`apps/`)
- **Streamlit Apps** (`apps/streamlit_apps/`): Interactive web applications
- **Web Dashboard** (`apps/web_dashboard/`): FastAPI backend with frontend

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Takedaxz/Stock-Analysis.git
cd Stock-Analysis
```

2. **Install dependencies**:
```bash
pip install -r config/requirements.txt
```

3. **Set up environment variables** (for API keys):
```bash
cp src/sentiment_analysis/secret.env.template src/sentiment_analysis/secret.env
# Edit secret.env with your API keys
```

## 🚀 Quick Start

### Running Streamlit Apps
```bash
cd apps/streamlit_apps
streamlit run stock_app.py
streamlit run news_app.py
streamlit run financial_app.py
```

### Running Technical Analysis
```bash
cd src/technical_analysis/indicators
python main.py
```

### Running Sentiment Analysis
```bash
cd src/sentiment_analysis
jupyter notebook gemini_sentiment_analyzer.ipynb
```

## 📋 Requirements

The project uses a comprehensive set of Python libraries including:

### Core Data Science
- `pandas`, `numpy`, `scipy`, `scikit-learn`
- `matplotlib`, `plotly`, `seaborn`
- `jupyter`, `jupyterlab`

### Financial Analysis
- `yfinance` - Yahoo Finance API
- `ta` - Technical Analysis library
- `mplfinance` - Financial plotting
- `streamlit` - Web applications

### AI & NLP
- `openai`, `google-generativeai`, `mistralai`
- `transformers`, `sentence-transformers`
- `nltk`, `pythainlp` - Natural language processing

### Web Scraping & APIs
- `requests`, `aiohttp`, `selenium`
- `beautifulsoup4`, `newspaper3k`
- `cloudscraper`, `curl_cffi`

### Machine Learning
- `tensorflow`, `torch`, `keras`
- `lightning`, `torchmetrics`
- `wandb` - Experiment tracking

## 🔑 Configuration

### API Keys Required
- **OpenAI API** - For GPT-based sentiment analysis
- **Google Gemini API** - For Gemini sentiment analysis
- **Mistral AI API** - For Mistral sentiment analysis

### Environment Setup
Create a `.env` file or use the existing `secret.env` in the `src/sentiment_analysis/` directory.

## 📊 Data Sources

- **Yahoo Finance API** - Stock prices, financial data
- **News APIs** - Financial news and market updates
- **Financial websites** - Earnings, events, and market data
- **Market indices** - S&P 500, NASDAQ 100, SET50

## 📈 Data Flow

1. **Data Collection** → Raw data stored in `data/raw/`
2. **Processing** → Processed data in `data/processed/`
3. **Analysis** → Results in respective module directories
4. **Visualization** → Interactive apps in `apps/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions, please open an issue on GitHub or contact the maintainers.

---

**Note**: This project is for educational and research purposes. Always do your own research before making investment decisions.
