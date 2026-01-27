#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Toto Prediction Dashboard — Full Supabase Integration
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import io
import os

# -------------------------
# Supabase setup
# -------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------
# Session state
# -------------------------
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()

if "lstm_model" not in st.session_state:
    st.session_state.lstm_model = None

# -------------------------
# Optional ML libs
# -------------------------
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ---------- Page config ----------
st.set_page_config(page_title="Toto Prediction — Dark Pro", layout="wide", page_icon="🎰")
st.title("🎰 Toto Prediction — Dark Pro")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    num_draws = st.slider("Number of past draws to analyze", min_value=20, max_value=2000, value=300, step=10)
    show_animated = st.checkbox("Show animated trend", True)
    train_epochs = st.number_input("LSTM train epochs", min_value=1, max_value=600, value=200)
    batch_size = st.number_input("Batch size", min_value=8, max_value=512, value=64)
    train_ratio = st.slider("Train ratio", 0.5, 0.95, 0.85)
    window_size = st.number_input("LSTM window size", min_value=1, max_value=30, value=10)
    seed = st.number_input("Random seed", value=42)
    mc_samples = st.number_input("MC prediction passes", min_value=1, max_value=200, value=20)

# -------------------------
# Scraper — requests + BeautifulSoup
# -------------------------
def scrape_toto_latest():
    url = "https://en.lottolyzer.com/history/singapore/toto?page=1"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to fetch page: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.select("table tbody tr")
    draws = []

    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 4:
            draw_no = int(cols[0].text.strip())
            draw_date = cols[1].text.strip()
            winning_no = cols[2].text.strip()
            additional_no = cols[3].text.strip()
            draws.append({
                "draw_no": draw_no,
                "draw_date": draw_date,
                "winning_no": winning_no,
                "additional_no": int(additional_no) if additional_no else None
            })
    return draws

# -------------------------
# Update Supabase
# -------------------------
def update_supabase(draws):
    for draw in draws:
        supabase.table("toto_results").upsert(draw, on_conflict="draw_no").execute()

# -------------------------
# Load from Supabase
# -------------------------
@st.cache_data(ttl=300)
def load_data_from_supabase(limit=None):
    query = supabase.table("toto_results") \
        .select("draw_no, draw_date, winning_no, additional_no") \
        .order("draw_no", desc=False)
    if limit:
        query = query.limit(limit)
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

# -------------------------
# Scrape & update button
# -------------------------
if st.button("Scrape & Update Latest Draws"):
    with st.spinner("Fetching latest TOTO draws..."):
        latest_draws = scrape_toto_latest()
        if latest_draws:
            update_supabase(latest_draws)
            st.success(f"{len(latest_draws)} draws updated in Supabase!")
        else:
            st.warning("No draws found or failed to fetch page.")
    st.session_state['df'] = load_data_from_supabase()

# -------------------------
# Load data into session_state
# -------------------------
if st.session_state['df'].empty:
    st.session_state['df'] = load_data_from_supabase()

df = st.session_state['df']

if df.empty:
    st.error("No data found in Supabase. Please run the scraper first.")
    st.stop()

st.write(f"**Dataset:** {len(df)} draws loaded — last draw date: {df.iloc[-1]['Draw Date']}")

# -------------------------
# Frequency helper
# -------------------------
@st.cache_data
def frequency_dataframe(df, last_n):
    recent = df.tail(last_n).reset_index(drop=True)
    all_nums = []
    frames = []

    for i, row in recent.iterrows():
        nums = row['Winning'][:]
        if row['Additional No'] is not None:
            nums.append(row['Additional No'])
        all_nums.extend(nums)
        subset = []
        for j in range(i+1):
            s_row = recent.iloc[j]
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
    return freq_df, frames_df

# -------------------------
# Tabs
# -------------------------
tab = st.radio("Navigation", ["Trends", "Hot / Cold Numbers", "Machine Learning Prediction"], horizontal=True)

# -------------------------
# Trends Tab
# -------------------------
if tab == "Trends":
    st.header("TOTO Trends & Statistics")
    latest_date = df["Draw Date"].iloc[-1]
    latest_main = [int(x) for x in df["Winning No"].iloc[-1].split(",")]
    latest_add = int(df["Additional No"].iloc[-1])
    st.subheader("Latest TOTO Result")
    st.write(f"**Draw Date:** {latest_date}")
    st.write(f"**Main Numbers:** {latest_main}")
    st.write(f"**Additional Number:** {latest_add}")
    st.markdown("---")

    freq_df, frames_df = frequency_dataframe(df, num_draws)
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
        st.subheader("Top / Bottom")
        top6 = list(freq_df.sort_values('Count', ascending=False).head(6)['Number'])
        bottom6 = list(freq_df.sort_values('Count', ascending=True).head(6)['Number'])
        st.metric("Top 6 (most frequent)", ', '.join(map(str, top6)))
        st.metric("Bottom 6 (least frequent)", ', '.join(map(str, bottom6)))

# -------------------------
# Hot / Cold Tab
# -------------------------
elif tab == "Hot / Cold Numbers":
    st.header("Hot and Cold Numbers")
    window = st.slider("Hot/Cold window (recent draws)", min_value=10, max_value=500, value=100, step=10)
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
        st.subheader("Hot Numbers (recent surge)")
        st.write(hot)
    with col2:
        st.subheader("Cold Numbers (least overall)")
        st.write(cold)

# -------------------------
# Machine Learning Tab (simplified)
# -------------------------
elif tab == "Machine Learning Prediction":
    st.header("Machine Learning Prediction — LSTM (7 numbers)")
    st.markdown("Train LSTM on 7-number draws (6 main + 1 additional).")

    if not TF_AVAILABLE:
        st.warning("TensorFlow not installed. Install it (`pip install tensorflow`) to use LSTM features.")
    else:
        st.info("LSTM training code can be added here (same as your previous implementation).")

