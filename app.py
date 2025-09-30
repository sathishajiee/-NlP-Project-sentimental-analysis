# app.py
import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# Load saved model and vectorizer
# ----------------------
try:
    model = pickle.load(open(r"C:/Users/gunac/OneDrive/Desktop/real time sentimental analysis/sentiment_model.pkl", "rb"))
    vectorizer = pickle.load(open(r"C:/Users/gunac/OneDrive/Desktop/real time sentimental analysis/tfidf_vectorizer.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Error loading model or vectorizer: {e}")
    st.stop()

# ----------------------
# App Title
# ----------------------
st.title("🛒 Real-Time Sentiment Analysis Dashboard")
st.write("Analyze customer reviews from e-commerce or social media in real-time!")

# ----------------------
# Section 1: Single Review
# ----------------------
st.header("Single Review Prediction")
user_input = st.text_area("Enter a review:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]
        if prediction == 0:
            st.error("😡 Negative Sentiment")
        else:
            st.success("😊 Positive Sentiment")

# ----------------------
# Section 2: CSV Batch Prediction
# ----------------------
st.header("Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV file with a column containing reviews", type=["csv", "xlsx"])

if uploaded_file:
    # Handle both CSV and Excel
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Try to detect the review column automatically
    possible_cols = ["review", "reviews", "Review", "Reviews", "text", "Text", "comment", "Comment"]
    review_col = None
    for col in df.columns:
        if col.strip() in possible_cols:
            review_col = col
            break

    if review_col is None:
        st.error("❌ CSV must have a column with reviews (e.g., 'review', 'text', or 'comment').")
    else:
        # Predict sentiments
        transformed = vectorizer.transform(df[review_col].astype(str))
        df['sentiment'] = model.predict(transformed)
        df['sentiment_label'] = df['sentiment'].apply(lambda x: "Positive 😊" if x == 1 else "Negative 😡")

        st.success(f"✅ Predictions Done! Using column: {review_col}")
        st.dataframe(df.head(10))

        # ----------------------
        # Section 3: Sentiment Distribution
        # ----------------------
        st.subheader("📊 Sentiment Distribution")
        plt.figure(figsize=(5,3))
        sns.countplot(x='sentiment_label', data=df, palette="coolwarm")
        plt.title("Sentiment Count")
        st.pyplot(plt.gcf())
        plt.clf()

        # ----------------------
        # Section 4: WordClouds
        # ----------------------
        st.subheader("☁️ WordClouds")
        positive_text = " ".join(df[df['sentiment'] == 1][review_col].astype(str))
        negative_text = " ".join(df[df['sentiment'] == 0][review_col].astype(str))

        if positive_text.strip():
            st.write("**Positive Reviews WordCloud**")
            wc_pos = WordCloud(width=600, height=300, background_color='white').generate(positive_text)
            plt.imshow(wc_pos, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt.gcf())
            plt.clf()

        if negative_text.strip():
            st.write("**Negative Reviews WordCloud**")
            wc_neg = WordCloud(width=600, height=300, background_color='white').generate(negative_text)
            plt.imshow(wc_neg, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt.gcf())
            plt.clf()
