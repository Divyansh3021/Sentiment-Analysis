import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Data Loading and Preprocessing
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Convert sentiment to binary (0 for negative, 1 for positive)
    df['sentiment_label'] = (df['sentiment'] == 'positive').astype(int)
    return df

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

# 2. Tokenization and Padding
def tokenize_and_pad(texts, max_length, tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=10000)  # Limit vocabulary size
        tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequences, tokenizer

# 3. Model Definition
def create_model(vocab_size, embedding_dim, max_length):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 4. Training with visualization
def train_model(model, X_train, y_train, epochs, batch_size):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]
    )
    return history

# 5. Evaluation and Visualization
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred = (model.predict(X_test) >= 0.5).astype(int)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return loss, accuracy

# 6. Prediction function
def predict_sentiment(model, tokenizer, text, max_length):
    cleaned_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    return 'Positive' if prediction >= 0.5 else 'Negative', prediction

# 7. Main execution
if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('Dataset.csv')
    print("Dataset shape:", data.shape)
    print("\nSample review:\n", data['review'].iloc[0][:200], "...\n")
    
    # Preprocess all reviews
    print("Preprocessing reviews...")
    data['cleaned_text'] = data['review'].apply(preprocess_text)
    print("Sample cleaned review:\n", data['cleaned_text'].iloc[0][:200], "...\n")
    
    # Tokenize and pad sequences
    max_length = 300  # Increased due to longer reviews
    print("Tokenizing and padding sequences...")
    X, tokenizer = tokenize_and_pad(data['cleaned_text'], max_length)
    y = data['sentiment_label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    
    # Create and train model
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    print("\nCreating model...")
    model = create_model(vocab_size, embedding_dim, max_length)
    model.summary()
    
    print("\nTraining model...")
    history = train_model(model, X_train, y_train, epochs=15, batch_size=32)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    loss, accuracy = evaluate_model(model, X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")

    # Save model and tokenizer
    print("\nSaving model and tokenizer...")
    model.save('sentiment_analysis_model.h5')
    import pickle
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Example prediction
    sample_text = data['review'].iloc[0]  # Use the first review as an example
    sentiment, confidence = predict_sentiment(model, tokenizer, sample_text, max_length)
    print(f"\nSample Prediction:")
    print(f"Text: {sample_text[:200]}...")
    print(f"Predicted Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.4f}")