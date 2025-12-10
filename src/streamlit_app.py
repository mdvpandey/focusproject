"""
streamlit_app.py
Simple Streamlit demo that loads model and shows focus score from a CSV of window features.
Usage:
    streamlit run src/streamlit_app.py
"""
import streamlit as st, pandas as pd, joblib, os

st.set_page_config(page_title="Focus Detection Demo", layout="centered")
st.title("Focus & Distraction Detection - Demo Dashboard")

uploaded = st.file_uploader("Upload window_features CSV (from window_aggregator)", type=['csv'])
model_file = st.file_uploader("Upload trained model (joblib)", type=['joblib','pkl'])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview of windows:", df.head(10))
    if model_file is not None:
        model = joblib.load(model_file)
        feature_cols = [c for c in df.columns if c not in ('window_id','label')]
        preds = model.predict(df[feature_cols].fillna(0))
        df['predicted_focus'] = preds
        score = (df['predicted_focus'].sum() / len(df)) * 100
        st.metric("Focus Score (%)", f"{score:.1f}")
        st.line_chart(df['predicted_focus'])
        st.dataframe(df[['window_id','predicted_focus']])
    else:
        st.info("Upload a trained model to see predictions")
else:
    st.info("Upload window_features CSV (one row per time window)")