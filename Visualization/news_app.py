# Visualization/news_app.py
import streamlit as st
import datetime
import pandas as pd
import sys
import os
import pytz

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
    st.title("News Summary and Sentiment Analysis")

    #st.markdown(f"Passed filepath: `{filepath}`")

    df = None

    if filepath is None:
        # st.error("No data file specified. Please provide a filepath.")
        try:
            default_path = os.path.join(os.path.dirname(__file__), "..", "CompletePipeline", "Data", "Gemini_news_2025-05-29_11-46.csv")
            #st.warning(f"Using default file path: `{default_path}`")
            df = pd.read_csv(default_path)
        except FileNotFoundError:
            #st.error(f"Default file not found at: `{default_path}`.")
            return
        except Exception as e:
            #st.error(f"Error loading default file: {e}")
            return
    else:
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            #st.error(f"File not found at: `{filepath}`. Please check the path relative to where you ran `streamlit run`.")
            return
        except Exception as e:
            #st.error(f"Error loading file from `{filepath}`: {e}")
            return

    if df is None or df.empty:
        st.warning("No data loaded. Please check the file path and content.")
        return

    df['publish_datetime'] = pd.to_datetime(df['publish_date'] + ' ' + df['publish_time'])
    df['publish_datetime'] = df['publish_datetime'] + pd.Timedelta(hours=7)

    df = df.sort_values(by='publish_datetime', ascending=False)
    
    score = calculate_sentiment_score(df)

    if score > 0.3:
        st.markdown(f"### Market Sentiment: <span style='color: green;'> **{score:.2f}**</span>", unsafe_allow_html=True)
    elif score < -0.3:
        st.markdown(f"### Market Sentiment: <span style='color: red;'> **{score:.2f}**</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"### Market Sentiment: <span style='color: gray;'> **{score:.2f}**</span>", unsafe_allow_html=True)
    
    bangkok_tz = pytz.timezone('Asia/Bangkok')
    utc_now = datetime.datetime.now(pytz.utc)
    bangkok_now = utc_now.astimezone(bangkok_tz)
    # st.markdown(f"### Overall Market Sentiment Score : **{score:.2f}**")
    st.markdown(f"Updated at: {bangkok_now.strftime('%Y-%m-%d %H:%M')} (UTC+7:00)")
    
    def display_news_card(row,number):
        sentiment_color = {
            'Positive': 'green',
            'Negative': 'red',
            'Neutral': 'gray'
        }
        
        st.markdown(f"---")
        
        st.subheader(f"{number}. [{row.title}]({row.url})")

        st.markdown(f"**Date Time**: {row.publish_datetime.strftime('%Y-%m-%d %H:%M')} (UTC+7:00)")
        
        st.markdown(f"**Sentiment:** <span style='color:{sentiment_color.get(row.sentiment, 'black')};'>**{row.sentiment}**</span>", unsafe_allow_html=True)
        
        st.markdown(f"**Importance:** {row.importance} / 5")
        
        escaped_summary = row.summary.replace("*", "\\*").replace("_", "\\_").replace("`", "\\`").replace("$", "\\$")
        st.markdown(f"**Summary:** {escaped_summary}")
        
        st.write(f"**Summary in Thai:** {row.translate}")
            

    for idx, row in enumerate(df.itertuples(), start=1):
        display_news_card(row, idx)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_filepath = sys.argv[1]
        app(filepath=data_filepath)
    else:
        app(filepath=None)