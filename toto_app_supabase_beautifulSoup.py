#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Toto Prediction Dashboard — Dark Theme, Supabase + BeautifulSoup

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os
import time
from datetime import datetime

# ---------- Supabase ----------
from supabase import create_client
import requests
from bs4 import BeautifulSoup

# ---------- Load from Streamlit Secrets ----------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- Session state ----------
if "lstm_model" not in st.session_state:
    st.session_state.lstm_model = None
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

# --- Optional ML libs ---
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

# ---------- Header ----------
col1, col2 = st.columns([0.12, 0.88])
with col1:
    logo_path = "dark_squirrel.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
with col2:
    st.markdown("<h1 style='color:#00e5ff'>🎰 Toto Prediction — Dark Pro</h1>", unsafe_allow_html=True)
    st.caption("Trends • Hot/Cold • Machine Learning (LSTM) — Click-to-refresh & PDF export")

st.write("---")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    num_draws = st.slider("Number of past draws to analyze", 20, 2000, 300, 10)
    show_animated = st.checkbox("Show animated trend", True)
    train_epochs = st.number_input("LSTM train epochs", 1, 600, 200)
    batch_size = st.number_input("Batch size", 8, 512, 64)
    train_ratio = st.slider("Train ratio", 0.5, 0.95, 0.85)
    window_size = st.number_input("LSTM window size", 1, 30, 10)
    seed = st.number_input("Random seed", value=42)
    mc_samples = st.number_input("MC prediction passes", 1, 200, 20)

# ---------- Disclaimer ----------
st.markdown("""
<div style="background-color:#fff3cd; padding:15px; border-left:6px solid #ffc107; border-radius:5px">
⚠️ **Disclaimer:** This TOTO app is for **fun and entertainment only**.  
Predictions are **not guaranteed**. We are **not responsible** for outcomes.
</div>
""", unsafe_allow_html=True)

# ---------- Load Data ----------
@st.cache_data(ttl=300)
def load_data_from_supabase(limit=None):
    query = supabase.table("toto_results").select("draw_no, draw_date, winning_no, additional_no").order("draw_no", desc=False)
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

# ---------- Scraper ----------
def scrape_latest_draws():
    url = "https://en.lottolyzer.com/history/singapore/toto?page=1"
    resp = requests.get(url)
    if resp.status_code != 200:
        st.error(f"Failed to fetch page: {resp.status_code}")
        return 0
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        st.error("Table not found on page")
        return 0
    rows = table.find("tbody").find_all("tr")
    added = 0
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 4:
            draw_no = cols[0].text.strip()
            draw_date = cols[1].text.strip()
            winning_no = cols[2].text.strip()
            additional_no = cols[3].text.strip()
            supabase.table("toto_results").upsert({
                "draw_no": draw_no,
                "draw_date": draw_date,
                "winning_no": winning_no,
                "additional_no": additional_no
            }).execute()
            added += 1
    st.session_state.df = load_data_from_supabase()
    return added

# ---------- Load or Refresh Data ----------
if 'df' not in st.session_state:
    st.session_state.df = load_data_from_supabase()

df = st.session_state['df']
st.write(f"**Dataset:** {len(df)} draws loaded — last draw date: {df.iloc[-1]['Draw Date']}")

if st.button("Scrape Latest Draws & Update Supabase"):
    added = scrape_latest_draws()
    st.success(f"{added} rows added/updated in Supabase!")

# ---------- Helper: Frequency ----------
@st.cache_data
def frequency_dataframe(df, last_n):
    recent = df.tail(last_n).reset_index(drop=True)
    all_nums, frames = [], []
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

# ---------- Tabs ----------
tab = st.radio("Navigation", ["Trends", "Hot / Cold Numbers", "Machine Learning Prediction"], horizontal=True)

# ---------------- Trends Tab ----------------
if tab == "Trends":
    st.header("TOTO Trends & Statistics")
    freq_df, frames_df = frequency_dataframe(df, num_draws)
    colA, colB = st.columns([2,1])
    with colA:
        st.subheader("Number Frequency (bar chart)")
        fig = px.bar(freq_df, x='Number', y='Count', title=f'Frequency (last {num_draws} draws)', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        if show_animated and not frames_df.empty:
            st.subheader("Animated frequency over draws")
            anim_fig = px.bar(frames_df, x='Number', y='Count', color='Number',
                              animation_frame='Frame', range_y=[0, frames_df['Count'].max()+1], template='plotly_dark')
            st.plotly_chart(anim_fig, use_container_width=True)
    with colB:
        st.subheader("Top / Bottom")
        top6 = list(freq_df.sort_values('Count', ascending=False).head(6)['Number'])
        bottom6 = list(freq_df.sort_values('Count', ascending=True).head(6)['Number'])
        st.metric("Top 6 (most frequent)", ', '.join(map(str, top6)))
        st.metric("Bottom 6 (least frequent)", ', '.join(map(str, bottom6)))

# ---------------- Hot / Cold Tab ----------------
elif tab == "Hot / Cold Numbers":
    st.header("Hot and Cold Numbers")
    window = st.slider("Hot/Cold window (recent draws)", 10, 500, 100, 10)
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

# ---------------- ML Tab ----------------
elif tab == "Machine Learning Prediction":
    st.header("Machine Learning Prediction — LSTM (7 numbers)")
    if not TF_AVAILABLE:
        st.warning("TensorFlow not installed. Install it (`pip install tensorflow`) to use LSTM features.")
    else:
        # --- ML preprocessing ---
        def draws_to_multihot(df_in):
            X = []
            for _, row in df_in.iterrows():
                v = np.zeros(49, dtype=np.float32)
                for n in row['Winning']:
                    v[int(n)-1] = 1.0
                if 'Additional No' in row and pd.notna(row['Additional No']):
                    v[int(row['Additional No'])-1] = 1.0
                X.append(v)
            return np.array(X)

        data_X = draws_to_multihot(df)
        if len(data_X) <= window_size:
            st.error("Not enough draws to build sequences. Reduce window size or add more data.")
        else:
            sequences, targets = [], []
            for i in range(len(data_X) - window_size):
                sequences.append(data_X[i:i+window_size])
                targets.append(data_X[i+window_size])
            sequences = np.array(sequences)
            targets = np.array(targets)
            st.write(f"Prepared {len(sequences)} sequences (window={window_size}) — features=49")

            model_path = "lstm_model.h5"

            def build_model(window_size, features=49):
                tf.random.set_seed(int(seed))
                model = keras.Sequential([
                    layers.Input(shape=(window_size, features)),
                    layers.LSTM(128),
                    layers.Dropout(0.2),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(features, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy')
                return model

            model = None
            if os.path.exists(model_path) and st.button("Load saved model"):
                with st.spinner("Loading model..."):
                    model = keras.models.load_model(model_path)
                st.success("Model loaded")

            if st.button("Train LSTM model"):
                if st.session_state.lstm_model is None:
                    st.session_state.lstm_model = build_model(window_size)
                model = st.session_state.lstm_model
                progress = st.progress(0)
                status = st.empty()
                start_time = time.time()
                history_logs = {"loss": [], "val_loss": []}
                for ep in range(train_epochs):
                    hist = model.fit(sequences, targets, epochs=1, batch_size=batch_size, validation_split=1-train_ratio, verbose=0)
                    loss = hist.history['loss'][-1]
                    val_loss = hist.history.get('val_loss', [None])[-1]
                    history_logs['loss'].append(loss)
                    history_logs['val_loss'].append(val_loss)
                    progress.progress(int((ep+1)/train_epochs*100))
                    status.text(f"Epoch {ep+1}/{train_epochs} — loss: {loss:.4f} val_loss: {val_loss}")
                model.save(model_path)
                st.success("Model trained & saved!")

