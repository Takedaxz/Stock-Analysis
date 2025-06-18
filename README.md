# Stock Analysis Project

A comprehensive stock market analysis platform that combines data collection, sentiment analysis, technical indicators, quantitative modeling, and interactive visualizations for informed investment decisions.

## Live Applications

- **News Sentiment Dashboard**: https://takedaxz-stock-news-sentiment.streamlit.app

## Project Structure

### Data Collection (`DataCollection/`)
Comprehensive data gathering from multiple sources:
- **`yfinanceAPI.ipynb`** - Yahoo Finance API integration for stock data
- **`ScrapingStockNews.ipynb`** - Stock-specific news scraping
- **`ScrapingMarketNews.ipynb`** - General market news collection
- **`ScrapingTrendingNews.ipynb`** - Trending financial news
- **`ScrapingFinancials.ipynb`** - Financial statement data extraction
- **`FinancialStatement.ipynb`** - Financial analysis and reporting
- **`EarningsCalendar.ipynb`** - Earnings calendar data collection
- **`EventCalendar.ipynb`** - Market events and calendar data
- **`Index.ipynb`** - Market index data (S&P 500, NASDAQ 100)

### Complete Pipeline (`CompletePipeline/`)
End-to-end data processing workflows:
- **`StockNews.ipynb`** - Automated stock news processing pipeline
- **`TrendingNews.ipynb`** - Trending news analysis pipeline
- **`Data/`** - Processed data files with timestamps

### Sentiment Analysis (`SentimentAnalysis/GPT/`)
AI-powered sentiment analysis using multiple models:
- **`GeminiNewsAnalysis.ipynb`** - Google Gemini sentiment analysis
- **`DeepSeekNewsAnalysis.ipynb`** - DeepSeek model analysis
- **`MistralNewsAnalysis.ipynb`** - Mistral AI sentiment analysis
- **`Output/`** - Sentiment analysis results by model

### Technical Analysis (`TechnicalAnalysis/`)
Advanced technical indicators and analysis:
- **`TradingDashboards.ipynb`** - Interactive trading dashboards
- **`HypothesisTesting.ipynb`** - Statistical hypothesis testing
- **`Indicators/`** - Comprehensive technical indicators:
  - **`main.py`** - Main technical analysis orchestrator
  - **`SMA.ipynb`** - Simple Moving Averages
  - **`EMA.ipynb`** - Exponential Moving Averages
  - **`RSI.ipynb`** - Relative Strength Index
  - **`MACD.ipynb`** - Moving Average Convergence Divergence
  - **`BollingerBands.ipynb`** - Bollinger Bands analysis
  - **`Stochastic_Oscillator.ipynb`** - Stochastic oscillator
  - **`Momentum.ipynb`** - Momentum indicators
  - **`Volume.ipynb`** - Volume analysis
  - **`SupportResistance.ipynb`** - Support and resistance levels
  - **`Ichimoku_Cloud.ipynb`** - Ichimoku cloud analysis
  - **`Elliott_Wave.ipynb`** - Elliott Wave theory
  - **`Chart_Patterns.ipynb`** - Chart pattern recognition

### Quantitative Analysis (`Quantitative/`)
Machine learning and statistical modeling:
- **`VixSET50.ipynb`** - VIX and SET50 volatility analysis
- **`Basic/`** - Basic quantitative models:
  - **`BasicPricePredicted.ipynb`** - Price prediction models
  - **`K-Means.ipynb`** - K-means clustering analysis
  - **`YuantaML1.ipynb`** - Yuanta machine learning models
  - **`YusantaML2.ipynb`** - Additional Yuanta ML models

### Visualization (`Visualization/`)
Interactive web applications and charts:
- **`stock_app.py`** - Stock analysis Streamlit app
- **`news_app.py`** - News sentiment Streamlit app
- **`financial_app.py`** - Financial data Streamlit app
- **`EnglishToThai.ipynb`** - Language translation utilities
- **`ReturnCalculation/`** - Return analysis:
  - **`DCA.ipynb`** - Dollar Cost Averaging analysis
  - **`Dividend.ipynb`** - Dividend analysis

### Core Analysis Files
- **`CandleStick.ipynb`** - Candlestick chart analysis
- **`EDA.ipynb`** - Exploratory Data Analysis

## Features

### Data Collection
- **Multi-source data gathering** from Yahoo Finance, news APIs, and financial websites
- **Automated scraping** of stock news, market news, and trending financial content
- **Financial statement analysis** with automated data extraction
- **Earnings and event calendar** tracking
- **Real-time market data** integration

### Sentiment Analysis
- **Multi-model AI analysis** using Gemini, DeepSeek, and Mistral
- **News sentiment scoring** with importance weighting
- **Automated sentiment classification** (Positive/Negative/Neutral)
- **Thai language support** with translation capabilities

### Technical Analysis
- **Comprehensive indicator suite** including RSI, MACD, Bollinger Bands, etc.
- **Advanced pattern recognition** for chart patterns and Elliott Waves
- **Support and resistance** level identification
- **Volume analysis** and momentum indicators
- **Interactive dashboards** for real-time analysis

### Quantitative Modeling
- **Machine learning models** for price prediction
- **Clustering analysis** for market segmentation
- **Volatility analysis** using VIX and market indices
- **Statistical hypothesis testing** for trading strategies

### Visualization & Apps
- **Streamlit web applications** for interactive analysis
- **Real-time data visualization** with Plotly and Matplotlib
- **Candlestick charts** with technical indicators
- **Sentiment dashboards** with live updates
- **Return calculation tools** for investment analysis

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Takedaxz/Stock-Analysis.git
cd Stock-Analysis
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (for API keys):
```bash
cp SentimentAnalysis/GPT/secret.env.example SentimentAnalysis/GPT/secret.env
# Edit secret.env with your API keys
```

## Requirements

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

## Usage Examples

### Running the Stock Analysis App
```bash
cd Visualization
streamlit run stock_app.py
```

### Running News Sentiment Analysis
```bash
cd SentimentAnalysis/GPT
jupyter notebook GeminiNewsAnalysis.ipynb
```

### Technical Analysis
```bash
cd TechnicalAnalysis/Indicators
python main.py
```

## Data Sources

- **Yahoo Finance API** - Stock prices, financial data
- **News APIs** - Financial news and market updates
- **Financial websites** - Earnings, events, and market data
- **Market indices** - S&P 500, NASDAQ 100, SET50

## Configuration

### API Keys Required
- **OpenAI API** - For GPT-based sentiment analysis
- **Google Gemini API** - For Gemini sentiment analysis
- **Mistral AI API** - For Mistral sentiment analysis

### Environment Setup
Create a `.env` file or use the existing `secret.env` in the SentimentAnalysis/GPT directory.

## Output Formats

- **CSV files** - Structured data with timestamps
- **JSON files** - API responses and configuration
- **Interactive charts** - Plotly and Matplotlib visualizations
- **Streamlit apps** - Web-based dashboards
- **Jupyter notebooks** - Analysis and documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with data source terms of service and API usage limits.

## Links

- **Live Demo**: https://takedaxz-stock-news-sentiment.streamlit.app
- **Documentation**: See individual notebook files for detailed analysis
- **Data Sources**: Yahoo Finance, Financial News APIs

---

**Note**: This project is designed for educational purposes and should not be used as the sole basis for investment decisions. Always conduct thorough research and consider consulting with financial advisors.
