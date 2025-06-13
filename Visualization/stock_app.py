# Visualization/stock_app.py
import streamlit as st
import datetime
import pandas as pd
import sys
import os
import pytz
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import glob

def get_latest_news_file(data_dir):
    pattern = os.path.join(data_dir, "*.csv")
    files = glob.glob(pattern)
    
    # Filter out files containing 'news' in their name
    files = [f for f in files if 'news' not in f.lower()]
    
    if not files:
        return None
        
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def calculate_sentiment_score(df):
    sentiment_scores = {
        'Positive': 1,
        'Negative': -1,
        'Neutral': 0
    }
    
    df['sentiment_score'] = df['sentiment'].map(sentiment_scores)
    
    df['weighted_score'] = df['sentiment_score'] * df['importance'].astype(int)
    
    total_importance = df['importance'].astype(int).sum()
    
    if total_importance == 0:
        return 0
    
    sentiment_score = df['weighted_score'].sum() / total_importance
    return sentiment_score

def app(filepath=None):

    st.set_page_config(layout="wide")

    df = None
    
    if filepath is None:
        #st.error("No data file specified. Please provide a filepath.")
        try:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "CompletePipeline", "Data")
            default_path = get_latest_news_file(data_dir)
            
            if default_path is None:
                st.error("No news data files found in the data directory.")
                return
                
            DEFAULT_PATH = default_path
            df = pd.read_csv(default_path)
        except Exception as e:
            st.error(f"Error loading default file: {e}")
            return
    else:
        try:
            df = pd.read_csv(filepath)
            DEFAULT_PATH = filepath
        except FileNotFoundError:
            #st.error(f"File not found at: `{filepath}`. Please check the path relative to where you ran `streamlit run`.")
            return
        except Exception as e:
            #st.error(f"Error loading file from `{filepath}`: {e}")
            return

    if df is None or df.empty:
        st.warning("No data loaded. Please check the file path and content.")
        return
    
    st.title(f"{df.ticker[0]} News Summary and Sentiment")
    try:
        stock = yf.download(tickers=df.ticker[0], period="3mo",interval="1d", progress=False, multi_level_index=False)
        
        fig = px.line(
            stock,
            y="Close",
            title=f"{df.ticker[0]} Price (Last 3 Months)",
            labels={"Close": "Price (USD)", "Datetime": "Datetime (UTC+0:00)"},
        )

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=400,
            width=800,
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load stock data. Error: {e}")
    
    df['publish_datetime'] = pd.to_datetime(df['publish_date'] + ' ' + df['publish_time'])
    df['publish_datetime'] = df['publish_datetime'] + pd.Timedelta(hours=7)

    df = df.sort_values(by='publish_datetime', ascending=False)
    
    score = calculate_sentiment_score(df)

    if score > 0.3:
        st.markdown(f"### Overall Market Sentiment: <span style='color: green;'> **{score:.2f}**</span>", unsafe_allow_html=True)
    elif score < -0.3:
        st.markdown(f"### Overall Market Sentiment: <span style='color: red;'> **{score:.2f}**</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"### Overall Market Sentiment: <span style='color: gray;'> **{score:.2f}**</span>", unsafe_allow_html=True)
    
    time=DEFAULT_PATH.split("/")[-1].split("_")[-2]+" "+DEFAULT_PATH.split("/")[-1].split("_")[-1].split(".")[0].replace("-", ":")
    st.markdown(f"Updated at: {time} (UTC+7:00)")
    
    def display_news_card(row,number):
        sentiment_color = {
            'Positive': 'green',
            'Negative': 'red',
            'Neutral': 'gray'
        }
        
        st.markdown(f"---")

        title = row.title.replace("*", "\\*").replace("_", "\\_").replace("`", "\\`").replace("$", "\\$")
        st.subheader(f"{number}. [{title}]({row.url})")

        st.markdown(f"**Date Time**: {row.publish_datetime.strftime('%Y-%m-%d %H:%M')} (UTC+7:00)")
        
        st.markdown(f"**Sentiment:** <span style='color:{sentiment_color.get(row.sentiment, 'black')};'>**{row.sentiment}**</span>", unsafe_allow_html=True)
        
        st.markdown(f"**Importance To Stock:** {row.importance} / 5")
        
        escaped_summary = row.summary.replace("*", "\\*").replace("_", "\\_").replace("`", "\\`").replace("$", "\\$")
        st.markdown(f"**Summary:** {escaped_summary}")
        
        escaped_summary_th = row.translate.replace("*", "\\*").replace("_", "\\_").replace("`", "\\`").replace("$", "\\$")
        st.markdown(f"**Summary in Thai:** {escaped_summary_th}")
            

    for idx, row in enumerate(df.itertuples(), start=1):
        display_news_card(row, idx)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_filepath = sys.argv[1]
        app(filepath=data_filepath)
    else:
        app(filepath=None)