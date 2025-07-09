import streamlit as st
import yfinance as yf
import re

def parse_financial_analysis(response):
    sections = response.split('**')
    data={
        'การเติบโตของรายได้และกำไร':sections[2].strip().replace('**', '').replace('\n', ' '),
        'อัตราส่วนทางการเงิน':sections[4].strip().replace('**', '').replace('\n', ' '),
        'กระแสเงินสด':sections[6].strip().replace('**', '').replace('\n', ' '),
        'สถานะทางการเงิน':sections[8].strip().replace('**', '').replace('\n', ' '),
        'จุดแข็ง':sections[10].strip().replace('**', '').replace('\n', ' ').replace('*', ' '),
        'จุดที่ควรปรับปรุง':sections[12].strip().replace('**', '').replace('\n', ' ').replace('*', ' '),
        'แนวโน้มและความยั่งยืน':sections[14].strip().replace('**', '').replace('\n', ' '),
        'ข้อเสนอแนะสำหรับนักลงทุน':sections[16].strip().replace('**', '').replace('\n', ' ')
    }
    return data

def display_financial_analysis(response, stock_name="STOCK"):
    """
    แสดงผลการวิเคราะห์ใน Streamlit
    """
    st.title(f"การวิเคราะห์งบการเงิน: {stock_name}")

    df=yf.download(stock_name, period='1y')
    st.write(f"Current price: {float(df['Close'].iloc[-1]):.2f}")
    st.line_chart(df['Close'])
    
    # แยกข้อมูล
    sections = parse_financial_analysis(response)

    # # แสดงผลแต่ละส่วน
    col1, col2 = st.columns(2)
    
    with col1:
        if 'การเติบโตของรายได้และกำไร' in sections:
            st.header("📈 การเติบโตของรายได้และกำไร")
            st.write(sections['การเติบโตของรายได้และกำไร'])
        
        if 'อัตราส่วนทางการเงิน' in sections:
            st.header("📊 อัตราส่วนทางการเงิน")
            st.write(sections['อัตราส่วนทางการเงิน'])
        
        if 'กระแสเงินสด' in sections:
            st.header("💰 กระแสเงินสด")
            st.write(sections['กระแสเงินสด'])
        
        if 'สถานะทางการเงิน' in sections:
            st.header("🏦 สถานะทางการเงิน")
            st.write(sections['สถานะทางการเงิน'])
    
    with col2:
        if 'จุดแข็ง' in sections:
            st.header("💪 จุดแข็ง")
            st.success(sections['จุดแข็ง'])
        
        if 'จุดที่ควรปรับปรุง' in sections:
            st.header("⚠️ จุดที่ควรปรับปรุง")
            st.warning(sections['จุดที่ควรปรับปรุง'])
        
        if 'แนวโน้มและความยั่งยืน' in sections:
            st.header("🔮 แนวโน้มและความยั่งยืน")
            st.info(sections['แนวโน้มและความยั่งยืน'])
        
        if 'ข้อเสนอแนะสำหรับนักลงทุน' in sections:
            st.header("💡 ข้อเสนอแนะสำหรับนักลงทุน")
            st.info(sections['ข้อเสนอแนะสำหรับนักลงทุน'])

if __name__ == "__main__":
    file_path = '../../src/data_collection/data/financials_analysis/aapl_2025-06-11_16-30_analysis.txt' 
    with open(file_path, 'r', encoding='utf-8') as file:
        response = file.read()
    stock_name = file_path.split('/')[-1].split('_')[0].upper()
    display_financial_analysis(response, stock_name=stock_name)