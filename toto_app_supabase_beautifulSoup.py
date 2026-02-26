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

###################################################################################
# -------------------------
# Supabase Storage Config
# -------------------------
MODEL_BUCKET = "models"
MODEL_FILE = "lstm_model.h5"
model_path = "lstm_model.h5"

def upload_model_to_supabase():
    try:
        with open(model_path, "rb") as f:
            supabase.storage.from_(MODEL_BUCKET).upload(
                path=MODEL_FILE,
                file=f,
                file_options={"upsert": True}
            )
        st.success("Model uploaded to Supabase successfully")
    except Exception as e:
        st.error(f"Upload failed: {e}")

def download_model_from_supabase():
    try:
        data = supabase.storage.from_(MODEL_BUCKET).download(MODEL_FILE)
        with open(model_path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


# Initialize session state if missing
if "lstm_model" not in st.session_state:
    st.session_state.lstm_model = None

# Now safe to check
if st.session_state.lstm_model is None:
    if os.path.exists(model_path):
        st.session_state.lstm_model = keras.models.load_model(model_path)
    else:
        if download_model_from_supabase():
            st.session_state.lstm_model = keras.models.load_model(model_path)

##################################################################################################################
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
# @st.cache_data(ttl=300)
def load_data_from_supabase(limit=None):
    query = supabase.table("toto_results") \
        .select("draw_no, draw_date, winning_no, additional_no") \
        .order("draw_no", desc=True)
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

st.write(f"**Dataset:** {len(df)} draws loaded — last draw date: {df.iloc[0]['Draw Date']}")

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
    latest_date = df["Draw Date"].iloc[0]
    latest_main = [int(x) for x in df["Winning No"].iloc[0].split(",")]
    latest_add = int(df["Additional No"].iloc[0])
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
# Machine Learning Tab — LSTM Prediction
# -------------------------
elif tab == "Machine Learning Prediction":
    st.header("Machine Learning Prediction — LSTM (7 numbers)")
    st.markdown("Train LSTM on 7-number draws (6 main + 1 additional). Progress + PDF export supported.")

    st.write(f"Epochs: {train_epochs}")
    st.write(f"Batch size: {batch_size}")
    st.write(f"Window size: {window_size}")
    st.write(f"Train ratio: {train_ratio}")

    if not TF_AVAILABLE:
        st.warning("TensorFlow not installed. Install it (`pip install tensorflow`) to use LSTM features.")
    else:
        # --- Helper: convert draws to multihot encoding ---
        def draws_to_multihot(df_in):
            X = []
            for _, row in df_in.iterrows():
                v = np.zeros(49, dtype=np.float32)
                for n in row['Winning']:
                    v[int(n)-1] = 1.0
                if row['Additional No'] is not None:
                    v[int(row['Additional No'])-1] = 1.0
                X.append(v)
            return np.array(X)

        data_X = draws_to_multihot(df)
        if len(data_X) <= window_size:
            st.error("Not enough draws to build sequences. Reduce window size or add more data.")
        else:
            sequences = []
            targets = []
            for i in range(len(data_X) - window_size):
                sequences.append(data_X[i:i+window_size])
                targets.append(data_X[i+window_size])
            sequences = np.array(sequences)
            targets = np.array(targets)
            st.write(f"Prepared {len(sequences)} sequences (window={window_size}) — features=49")

            model_path = "lstm_model.h5"

            # --- Build model ---
            def build_model(window_size, features=49):
                tf.random.set_seed(seed)
                model = keras.Sequential([
                    layers.Input(shape=(window_size, features)),
                    layers.LSTM(128, return_sequences=False),
                    layers.Dropout(0.2),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(features, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy')
                return model

            model = None
            if os.path.exists(model_path):
                st.info("Saved model found on disk")
                if st.button("Load saved model"):
                    with st.spinner("Loading model..."):
                        model = keras.models.load_model(model_path)
                    st.success("Model loaded")

            # --- Train LSTM ---
            if st.button("Train LSTM model", key="train_lstm"):

                # CREATE MODEL ONLY IF NOT EXISTS
                if st.session_state.lstm_model is None:
                    st.session_state.lstm_model = build_model(window_size)
                model = st.session_state.lstm_model  # use session model

                progress = st.progress(0)
                status = st.empty()
                loss_chart = st.empty()
                start_time = time.time()
                history_logs = {"loss": [], "val_loss": []}

                val_split = 1.0 - float(train_ratio)

                for ep in range(train_epochs):
                    hist = model.fit(sequences, targets,
                                     epochs=1,
                                     batch_size=batch_size,
                                     validation_split=val_split,
                                     verbose=0)
                    loss = hist.history.get('loss', [None])[-1]
                    val_loss = hist.history.get('val_loss', [None])[-1]
                    history_logs['loss'].append(loss)
                    history_logs['val_loss'].append(val_loss)

                    percent = int(((ep + 1) / train_epochs) * 100)
                    progress.progress(percent)
                    elapsed = time.time() - start_time
                    avg_per_epoch = elapsed / (ep + 1)
                    remaining = avg_per_epoch * (train_epochs - (ep + 1))
                    status.text(f"Epoch {ep+1}/{train_epochs} — loss: {loss:.4f} val_loss: {val_loss:.4f} — ETA: {remaining:.1f}s")

                    loss_chart.line_chart({
                        "loss": history_logs['loss'],
                        "val_loss": history_logs['val_loss']
                    })
#########################################################################################################################################
                # model.save(model_path)
                model.save(model_path)
                upload_model_to_supabase()

                st.session_state.lstm_model = model
######################################################################################################################

                progress.progress(100)
                status.text(f"Training completed in {time.time() - start_time:.1f}s — model saved")
                st.success("Model training finished and saved")

            # --- Prediction ---
            st.markdown("### Predict next draw (LSTM)")

            last_n_for_priority = st.number_input("Number of recent draws to prioritize", min_value=5, max_value=50, value=10, step=1)

            if st.button("Predict next draw (LSTM)"):

                model = st.session_state.lstm_model

                if model is None:
                    st.error("No trained model available. Train the model first.")
                    st.stop()      

                # if model is None:
                #     if os.path.exists(model_path):
                #         with st.spinner("Loading saved model..."):
                #             model = keras.models.load_model(model_path)
                #     else:
                #         st.error("No trained model available. Train or load a model first.")
                #         model = None

                if model is not None:
                    last_seq = data_X[-window_size:]
                    inp = last_seq.reshape((1, window_size, 49)).astype(np.float32)

                    mc = int(mc_samples)
                    probs_accum = np.zeros(49, dtype=np.float64)
                    prog = st.progress(0)
                    status_p = st.empty()
                    t0 = time.time()

                    for i in range(mc):
                        pred = model(inp, training=True).numpy().reshape(-1)
                        probs_accum += pred
                        prog.progress(int(((i+1)/mc)*100))
                        elapsed = time.time() - t0
                        avg = elapsed / (i+1)
                        remaining = avg * (mc - (i+1))
                        status_p.text(f"MC pass {i+1}/{mc} — ETA: {remaining:.1f}s")

                    avg_probs = probs_accum / mc

                    # prioritize numbers from last N draws
                    recent_draws = df.tail(last_n_for_priority)
                    recent_numbers = set()
                    for _, row in recent_draws.iterrows():
                        recent_numbers.update(row['Winning'])
                        if row['Additional No'] is not None:
                            recent_numbers.add(row['Additional No'])

                    all_probs_sorted = [(i+1, avg_probs[i]) for i in range(49)]
                    all_probs_sorted.sort(key=lambda x: x[1], reverse=True)

                    top7_probs = []
                    for num, prob in all_probs_sorted:
                        if num in recent_numbers:
                            top7_probs.append((num, prob))
                        if len(top7_probs) == 7:
                            break
                    if len(top7_probs) < 7:
                        for num, prob in all_probs_sorted:
                            if num not in [x[0] for x in top7_probs]:
                                top7_probs.append((num, prob))
                            if len(top7_probs) == 7:
                                break

                    table_data = []
                    for num, prob in top7_probs:
                        table_data.append({
                            'Number': num,
                            'Prob': prob,
                            'Recent (last N draws)': 'Yes' if num in recent_numbers else 'No'
                        })
                    top7_idx = [x['Number'] for x in table_data]

                    status_p.text("Prediction done.")
                    prog.progress(100)
                    st.success(f"Predicted numbers (6 main + 1 additional): {top7_idx}")
                    st.table(pd.DataFrame(table_data).set_index('Number'))

                    # PDF export
                    if REPORTLAB_AVAILABLE:
                        try:
                            pdf_bytes = io.BytesIO()
                            c = canvas.Canvas(pdf_bytes, pagesize=letter)
                            text = c.beginText(40, 700)
                            text.setFont("Helvetica", 14)
                            text.textLine("Toto Prediction — LSTM")
                            text.textLine(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            text.textLine("")
                            text.textLine("Predicted numbers (6 main + additional):")
                            text.textLine(', '.join(map(str, list(top7_idx))))
                            c.drawText(text)
                            c.showPage()
                            c.save()
                            pdf_bytes.seek(0)
                            st.download_button("Download prediction as PDF", data=pdf_bytes, file_name="toto_prediction.pdf", mime='application/pdf')
                        except TypeError:
                            st.warning("PDF export temporarily unavailable.")


# import base64
# 
# def to_base64(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode()
# 
# icon192 = to_base64("icon-192.png")
# icon512 = to_base64("icon-512.png")
# icon180 = to_base64("icon-180.png")
# 
# print(icon192[:50])  # just to confirm
# 
