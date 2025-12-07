import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="App Review Intelligence",
    page_icon="📊",
    layout="wide"
)

# ============================================
# CLEANING FUNCTION
# ============================================
def clean_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return " ".join(text.split())

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    tfidf = pickle.load(open("tfidf_vectorizer.pkl","rb"))
    sentiment = pickle.load(open("sentiment_model.pkl","rb"))
    count_vec = pickle.load(open("count_vectorizer.pkl","rb"))
    lda = pickle.load(open("lda_model.pkl","rb"))
    topics = pickle.load(open("topic_labels.pkl","rb"))
    return tfidf, sentiment, count_vec, lda, topics

tfidf, sentiment_model, count_vec, lda_model, topic_labels = load_models()

# ============================================
# SENTIMENT + TOPIC PREDICTORS (CORRECTED)
# ============================================
def predict_sentiment_batch(texts):
    X = tfidf.transform(texts)
    
    preds = sentiment_model.predict(X)

    # True class order inside the model
    class_order = list(sentiment_model.classes_)

    try:
        probas = sentiment_model.predict_proba(X)
    except:
        decision = sentiment_model.decision_function(X)
        exp = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        probas = exp / exp.sum(axis=1, keepdims=True)

    return preds, probas, class_order


def predict_topics_batch(texts):
    X = count_vec.transform(texts)
    dist = lda_model.transform(X)
    idxs = dist.argmax(axis=1)
    labels = [list(topic_labels.values())[i] for i in idxs]
    return labels, dist

# ============================================
# UI
# ============================================
st.title("📊 App Review Intelligence – Batch Analyzer")
st.write("Upload a CSV file with a **column named 'content'** containing app reviews.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    if "content" not in df.columns:
        st.error("CSV must contain a column named **content**.")
        st.stop()

    st.success("CSV Loaded Successfully!")
    st.write(df.head())

    # Clean text
    df["clean_text"] = df["content"].apply(clean_text)

    # ------------------- SENTIMENT -------------------
    sentiments, sentiment_probs, class_order = predict_sentiment_batch(df["clean_text"])
    df["sentiment"] = sentiments

    # ------------------- TOPICS ----------------------
    topic_names, topic_dists = predict_topics_batch(df["clean_text"])
    df["topic"] = topic_names

    # ============================================
    # SENTIMENT CHART (correct ordering)
    # ============================================
    st.markdown("## 📈 Sentiment Distribution")

    # Ensure counts follow real model order
    sent_counts = df["sentiment"].value_counts().reindex(class_order, fill_value=0)
    st.bar_chart(sent_counts)

    # ============================================
    # TOPIC CHART
    # ============================================
    st.markdown("## 🗂 Topic Distribution")
    topic_counts = df["topic"].value_counts()
    st.bar_chart(topic_counts)

    # ============================================
    # SAMPLE OUTPUT
    # ============================================
    st.markdown("## 🔍 Sample Predictions")
    st.write(df[["content","sentiment","topic"]].head(10))

    # ============================================
    # DOWNLOAD BUTTON
    # ============================================
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results CSV",
        data=csv,
        file_name="review_analysis_output.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("CIS 508 – Machine Learning in Business | Streamlit Demo Interface")
