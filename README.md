# 📰 Fake News Detection Engine

A clean, interactive Streamlit web application that uses a Machine Learning pipeline to analyze news articles and predict whether they are Real or Fake with **98% accuracy**.

## 🚀 Features
* **Machine Learning Pipeline:** Powered by a Multinomial Naive Bayes model and a TF-IDF Vectorizer.
* **Streamlit Web Interface:** A professional, responsive layout including a sidebar with model metrics and live text word counters.
* **Server-Side Optimization:** Uses memory caching (`@st.cache_resource`) to load the 187MB model file exactly once, ensuring lightning-fast multi-user performance.

## 📁 Project Directory Setup
Ensure your local project folder is arranged like this before running or deploying:
```text
├── fake_news_pipeline.joblib  # 187MB trained model pipeline file
├── app.py                     # The Streamlit application script
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation

![Dashboard Screenshot](/image.png)

[Dataset Link](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)