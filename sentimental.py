import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================
# Load trained model
# ==========================
model = load_model("model.keras")

# ==========================
# Load tokenizer
# ==========================
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Max sequence length
max_len = 120

# Sentiment labels
sentiment_labels = ["Negative", "Neutral", "Positive"]


# ==========================
# Prediction Function
# ==========================
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)

    prediction = model.predict(padded)

    label = np.argmax(prediction)

    return sentiment_labels[label]


# ==========================
# STREAMLIT WEB UI
# ==========================

st.set_page_config(page_title="Laptop Sentiment AI", page_icon="💻")

st.title("💻 Laptop Review Sentiment Analyzer")

st.write("Enter a laptop review and detect its sentiment using AI.")

text = st.text_area("Enter Laptop Review")

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter a review")
    else:
        result = predict_sentiment(text)

        if result == "Positive":
            st.success("😊 Positive Sentiment")
        elif result == "Negative":
            st.error("😞 Negative Sentiment")
        else:
            st.info("😐 Neutral Sentiment")
            
