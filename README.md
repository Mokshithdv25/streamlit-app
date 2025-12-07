ChatGPT App Review Intelligence Dashboard

Machine Learning Term Project — CIS 508 (Arizona State University)

Overview

The ChatGPT mobile app receives 860,000+ reviews, but product teams typically analyze only 10–15% manually due to time constraints. This leads to:

Missed critical issues

Slow feature prioritization

Delayed responses to user pain points

This project builds an end-to-end ML pipeline that automates sentiment analysis and topic extraction to help product teams understand user feedback at scale.

What This System Does
1. Sentiment Classification (Supervised ML)

Classifies user reviews into:

Positive

Neutral

Negative

Model: TF-IDF + Linear SVM
Accuracy on held-out test set: ≈ 60% (expected for noisy App Store data)

2. Topic Modeling (Unsupervised ML)

Runs LDA on all ≤3-star reviews to detect dominant themes.
Discovered topics include:

Response Quality & Accuracy

Subscription & Billing Issues

Performance & Technical Bugs

Feature Limitations

Customer Support Problems

3. Interactive Streamlit Dashboard

The deployed app lets users:

Paste any review

Get predicted sentiment

Get dominant topic

See model explanations

Use it as a PM intelligence tool

Deployed App:
<INSERT STREAMLIT URL HERE>

Repository Structure
chatgpt-review-intelligence-app/
│
├── app.py                 # Streamlit application
├── requirements.txt       # Dependencies for Streamlit Cloud
└── models/                # Exported ML models
    ├── tfidf_vectorizer.pkl
    ├── sentiment_model.pkl
    ├── count_vectorizer.pkl
    ├── lda_model.pkl
    └── topic_labels.pkl

Model Development Workflow
1. Data Preparation

878k reviews loaded from Databricks

URL removal, punctuation stripping, normalization

Removal of short/empty reviews

Deduplication

Balanced sampling (~18k per sentiment class)

2. Feature Engineering

TF-IDF with 5,000 features (unigrams + bigrams)

CountVectorizer (1,500 features) for LDA topics

3. Models

Logistic Regression baseline

Linear SVM (final model)

LDA topic model (5 topics)

All experiments tracked in MLflow

4. Deployment

The app loads each saved artifact:

TF-IDF vectorizer

SVM sentiment classifier

LDA model

CountVectorizer

Human-interpreted topic labels

How to Run Locally

Clone the repo:

git clone https://github.com/<your-username>/chatgpt-review-intelligence-app.git
cd chatgpt-review-intelligence-app


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py

Deployment Instructions (Streamlit Cloud)

Push this repo to GitHub

Go to: https://share.streamlit.io

Choose New App

Select this repo → main branch → app.py

Deploy

The app will build automatically and provide a public URL.

Business Impact Summary

90% reduction in manual review processing time

$50,000 annual savings at PM salary levels

100% coverage of all app reviews

Real-time monitoring of issues after releases

Prioritization guidance for product roadmaps

Technologies Used

Python

Scikit-Learn

Latent Dirichlet Allocation (LDA)

TF-IDF

Pandas / NumPy

Streamlit

Databricks + MLflow

Author

Mokshith Diggenahalli Vasanth Kumar
CIS 508 — Machine Learning in Business
Arizona State University

License

This project is released for academic and educational use.
