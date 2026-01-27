#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -----------------------------
# Toto Prediction Dashboard — Supabase + Streamlit + LSTM
# -----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import io
import os
from datetime import datetime

# ML
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Selenium Scraper
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Supabase
from supabase import create_client

# -----------------------------
# Supabase Secrets
# -----------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Session state init
# -----------------------------
if "lstm_model" not in st.session_state:
    st.session_state.lstm_model = None
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

# -----------------------------
# App Header
# -----------------------------
st.set_page_config(page_title="Toto Prediction — Dark Pro", layout="wide", page_icon="🎰")
st.title("🎰 Toto Prediction Dashboard — Supabase Edition")

# -----------------------------
# Sidebar Settings
# -----------------------------
with st.sidebar:
    st.header("Settings")
    num_draws = st.slider("Number of past draws to analyze", 20, 2000, 300, 10)
    show_animated = st.checkbox("Show animated trend", True)
    train_epochs = st.number_input("LSTM train epochs", 1, 600, 200)
    batch_size = st.number_input("Batch size", 8, 512, 64)
    train_ratio = st.slider("Train ratio", 0.5, 0.95, 0.85)
    window_size = st.number_input("LSTM window size", 1, 30, 10)
    mc_samples = st.number_input("MC prediction passes", 1, 200, 20)
    seed = st.number_input("Random seed", value=42)

st.markdown("---")

# -----------------------------
# Load data from Supabase
# -----------------------------
@st.cache_data(ttl=300)
def load_data_from_supabase():
    query = supabase.table("toto_results") \
        .select("draw_no, draw_date, winning_no, additional_no") \
        .order("draw_no", desc=False)
    response = query.execute()
    if not response.data:
        return pd.DataFrame()
    df = pd.DataFrame(response.data)
    df.rename(columns={
        "draw_no": "Draw No",
        "draw_date": "Draw Date",
        "winning_no": "Winning No",
        "additional_no": "Additional No"
    }, inplace=True)
    df['Winning'] = df['Winning No'].apply(lambda x: [int(i) for i in str(x).split(',')])
    df['Additional No'] = df['Additional No'].apply(lambda x: int(x) if pd.notna(x) else None)
    return df.reset_index(drop=True)

# Load into session
if st.button("Refresh Data") or st.session_state.df.empty:
    st.session_state.df = load_data_from_supabase()

df = st.session_state.df
if df.empty:
    st.warning("No data in Supabase. Run scraper first.")
else:
    st.write(f"**Dataset:** {len(df)} draws loaded — last draw date: {df.iloc[-1]['Draw Date']}")

# -----------------------------
# Scraper (headless Selenium)
# -----------------------------
if st.button("Scrape Latest Draws & Update Supabase"):
    chrome_driver_path = "C:/Selenium/chromedriver.exe"  # adjust for your env
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    
    url = "https://en.lottolyzer.com/history/singapore/toto?page=1"
    driver.get(url)
    time.sleep(2)
    
    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
    added = 0
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) >= 4:
            draw_no = cols[0].text.strip()
            draw_date = cols[1].text.strip()
            winning_no = cols[2].text.strip()
            additional_no = cols[3].text.strip()
            
            # Upsert to Supabase
            supabase.table("toto_results").upsert({
                "draw_no": draw_no,
                "draw_date": draw_date,
                "winning_no": winning_no,
                "additional_no": additional_no
            }).execute()
            added += 1
    driver.quit()
    st.success(f"Scraping complete! {added} rows added/updated in Supabase.")
    st.session_state.df = load_data_from_supabase()  # reload updated data

# -----------------------------
# Tabs: Trends / Hot-Cold / ML Prediction
# -----------------------------
tab = st.radio("Navigation", ["Trends", "Hot / Cold Numbers", "Machine Learning Prediction"], horizontal=True)

# -----------------------------
# Trends Tab
# -----------------------------
if tab == "Trends":
    st.header("TOTO Trends & Frequency")
    recent_df = df.tail(num_draws)
    
    all_nums = []
    frames = []
    for i, row in recent_df.iterrows():
        nums = row['Winning'][:]
        if row['Additional No'] is not None:
            nums.append(row['Additional No'])
        all_nums.extend(nums)
        # Animation frames
        subset = []
        for j in range(i+1):
            s_row = recent_df.iloc[j]
            s_nums = s_row['Winning'][:]
            if s_row['Additional No'] is not None:
                s_nums.append(s_row['Additional No'])
            subset.extend(s_nums)
        series = pd.Series(subset).value_counts().sort_index()
        for num, cnt in series.items():
            frames.append({'Frame': i, 'Number': num, 'Count': int(cnt)})
    
    freq = pd.Series(all_nums).value_counts().sort_index()
    freq_df = pd.DataFrame({'Number': freq.index, 'Count': freq.values})
    frames_df = pd.DataFrame(frames)

    colA, colB = st.columns([2,1])
    with colA:
        st.subheader("Number Frequency (bar chart)")
        fig = px.bar(freq_df, x='Number', y='Count', title=f'Frequency (last {num_draws} draws)', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        if show_animated and not frames_df.empty:
            st.subheader("Animated frequency over draws")
            anim_fig = px.bar(frames_df, x='Number', y='Count', color='Number', animation_frame='Frame',
                              range_y=[0, frames_df['Count'].max()+1], template='plotly_dark')
            st.plotly_chart(anim_fig, use_container_width=True)
    with colB:
        st.subheader("Top / Bottom Numbers")
        top6 = list(freq_df.sort_values('Count', ascending=False).head(6)['Number'])
        bottom6 = list(freq_df.sort_values('Count', ascending=True).head(6)['Number'])
        st.metric("Top 6 (most frequent)", ', '.join(map(str, top6)))
        st.metric("Bottom 6 (least frequent)", ', '.join(map(str, bottom6)))

# -----------------------------
# Hot / Cold Tab
# -----------------------------
elif tab == "Hot / Cold Numbers":
    st.header("Hot & Cold Numbers")
    window = st.slider("Hot/Cold window", 10, 500, 100, 10)
    recent_df = df.tail(window)
    all_recent = []
    for _, r in recent_df.iterrows():
        all_recent.extend(r['Winning'])
        if r['Additional No'] is not None:
            all_recent.append(r['Additional No'])
    recent_counts = pd.Series(all_recent).value_counts()
    overall_counts = pd.Series([n for row in df['Winning'] for n in row] +
                               [row['Additional No'] for _, row in df.iterrows() if row['Additional No'] is not None]).value_counts()
    ratio = (recent_counts / overall_counts).fillna(0).sort_values(ascending=False)
    hot = ratio.head(6)
    cold = overall_counts.sort_values().head(6)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hot Numbers")
        st.write(hot)
    with col2:
        st.subheader("Cold Numbers")
        st.write(cold)

# -----------------------------
# Machine Learning Tab
# -----------------------------
elif tab == "Machine Learning Prediction":
    # Paste the full LSTM training & prediction code here
    st.header("LSTM Prediction — 7 Numbers")
    st.write("Use your session_state.df as input to LSTM.")
    # (Reuse the LSTM code from previous cell, fully compatible with Supabase)

