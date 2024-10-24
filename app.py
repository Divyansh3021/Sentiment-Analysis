import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import plotly.graph_objects as go
import plotly.express as px
import time

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load the saved model and tokenizer
@st.cache_resource
def load_saved_model():
    model = load_model('sentiment_analysis_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<br\s*/?>|<[^>]+>', ' ', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def predict_sentiment(text, model, tokenizer, max_length=300):
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    # Convert to sequence and pad
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Make prediction
    prediction = model.predict(padded)[0][0]
    return prediction

def create_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightcoral"},
                {'range': [33, 66], 'color': "khaki"},
                {'range': [66, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Download NLTK data
    download_nltk_data()
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_saved_model()
    except:
        st.error("Error: Could not load the model or tokenizer. Please make sure the files exist in the correct location.")
        return

    # Set page configuration
    # st.set_page_config(
    #     page_title="Movie Review Sentiment Analysis",
    #     page_icon="🎬",
    #     layout="wide"
    # )

    # Main title
    st.title("🎬 Movie Review Sentiment Analysis")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This application uses a deep learning model to analyze the sentiment of "
        "movie reviews. The model has been trained on thousands of movie reviews "
        "and can predict whether a review is positive or negative."
    )
    
    st.sidebar.header("How to use")
    st.sidebar.info(
        "1. Enter your movie review in the text box\n"
        "2. Click 'Analyze Sentiment'\n"
        "3. View the sentiment prediction and confidence score"
    )

    # Main content
    st.write(
        "Enter a movie review below and the model will analyze its sentiment. "
        "The model will predict whether the review is positive or negative and "
        "provide a confidence score."
    )

    # Text input
    review_text = st.text_area(
        "Enter your movie review:",
        height=200,
        placeholder="Type or paste your movie review here..."
    )

    # Analysis button
    if st.button("Analyze Sentiment"):
        if review_text.strip() == "":
            st.warning("Please enter a review first.")
        else:
            # Show spinner while processing
            with st.spinner("Analyzing sentiment..."):
                # Add slight delay for better UX
                time.sleep(1)
                
                # Get prediction
                prediction_score = predict_sentiment(review_text, model, tokenizer)
                
                # Create two columns for results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display sentiment result
                    sentiment = "Positive" if prediction_score >= 0.5 else "Negative"
                    st.subheader("Predicted Sentiment")
                    sentiment_color = "green" if sentiment == "Positive" else "red"
                    st.markdown(f"<h1 style='color: {sentiment_color};'>{sentiment}</h1>", 
                              unsafe_allow_html=True)
                
                with col2:
                    # Display confidence gauge
                    st.subheader("Confidence Score")
                    gauge_chart = create_gauge_chart(prediction_score)
                    st.plotly_chart(gauge_chart, use_container_width=True)
                
                # Show processed text
                with st.expander("View Preprocessed Text"):
                    st.write(preprocess_text(review_text))

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center;'>
            Created with ❤️ using Streamlit | Model: LSTM Neural Network
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()