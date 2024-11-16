import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the classifier model
with open("news_classifier_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

# Load the vectorizer
with open("vectorizer.pkl", "rb") as vec_file:
    loaded_vectorizer = pickle.load(vec_file)

# Streamlit UI
st.title("Real vs Fake News Classifier")
st.write("This application classifies news articles as **Real** or **Fake** based on the provided text.")
news_text = st.text_area("Enter the news article text here:", height=200)

if st.button("Classify News"):
    if news_text.strip():
        try:
            transformed_text = loaded_vectorizer.transform([news_text])
            prediction = loaded_model.predict(transformed_text)
            label = "Real" if prediction[0] == 1 else "Fake"
            st.success(f"The news article is classified as: **{label}**")
        except Exception as e:
            st.error(f"An error occurred: s{str(e)}")
    else:
        st.warning("Please enter some text to classify.")
