#streamlit run Visualizing/news_app.py
import streamlit as st
import pandas as pd
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

def app():
    st.set_page_config(layout="wide")
    st.title("News Summary and Sentiment Analysis")

    try:
        df = pd.read_csv('Visualizing/ForVisualize/Gemini_news_2025-05-28_01-11-58.csv')
    except FileNotFoundError:
        st.error("File not found.")
        return

    df['publish_datetime'] = pd.to_datetime(df['publish_date'] + ' ' + df['publish_time'])

    df = df.sort_values(by='publish_datetime', ascending=False)
    
    score = calculate_sentiment_score(df)
    
    st.markdown(f"### Overall Market Sentiment Score: **{score:.2f}**")
    
    def display_news_card(row,number):
        sentiment_color = {
            'Positive': 'green',
            'Negative': 'red',
            'Neutral': 'gray'
        }
        
        st.markdown(f"---")
        
        st.subheader(f"{number}. [{row.title}]({row.url})")

        st.markdown(f"**Date**: {row.publish_date}")
        st.markdown(f"**Time**: {row.publish_time}")
        
        st.markdown(f"**Sentiment:** <span style='color:{sentiment_color.get(row.sentiment, 'black')};'>**{row.sentiment}**</span>", unsafe_allow_html=True)
        
        st.markdown(f"**Importance:** {row.importance} / 5")
        
        escaped_summary = row.summary.replace("*", "\\*").replace("_", "\\_").replace("`", "\\`").replace("$", "\\$")
        st.markdown(f"**Summary:** {escaped_summary}")
        
        st.write(f"**Summary in Thai:** {row.translate}")
            

    for idx, row in enumerate(df.itertuples(), start=1):
        display_news_card(row, idx)

if __name__ == '__main__':
    app()