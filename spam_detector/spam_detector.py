# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import joblib
import os

# Download NLTK data at the start (quiet mode for Streamlit Cloud)
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

# Set page config
st.set_page_config(page_title="Email Spam Detector", page_icon="📧")

st.title("📧 Email Spam Detector")
st.write("This app uses Machine Learning to identify spam messages.")

# Text Preprocessing Function
stop_words = ENGLISH_STOP_WORDS

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Replace email addresses
    text = re.sub(r'\S+@\S+', ' emailaddr ', text)
    # Replace URLs
    text = re.sub(r'http[s]?://\S+', ' url ', text)
    text = re.sub(r'www\.\S+', ' url ', text)
    # Replace money symbols
    text = re.sub(r'£|\$', ' money ', text)
    # Replace phone numbers
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', ' phonenumber ', text)
    # Replace numbers
    text = re.sub(r'\d+', ' number ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize
    words = text.split()
    return ' '.join(words)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_models():
    """Load pre-trained models and vectorizer"""
    try:
        # Try to load existing models
        model_lr = joblib.load("logistic_model.pkl")
        model_nb = joblib.load("naive_bayes_model.pkl")
        vectorizer = joblib.load("tfidf.pkl")
        return model_lr, model_nb, vectorizer, True
    except:
        st.warning("⚠️ Pre-trained models not found. Training new models...")
        return None, None, None, False

# Load or train models
model_lr, model_nb, vectorizer, models_loaded = load_models()

# If models don't exist, train them
if not models_loaded:
    with st.spinner("Training models... This may take a minute."):
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        
        # Try different possible paths for the CSV file
        csv_paths = [
            "spam.csv",
            "Streamlit_app/spam.csv",
            "spam_detector/spam.csv"
        ]
        
        df = None
        for path in csv_paths:
            if os.path.exists(path):
                df = pd.read_csv(path, encoding='latin1')
                break
        
        if df is None:
            st.error("❌ Could not find spam.csv file. Please ensure it's in your repository.")
            st.stop()
        
        # Clean and prepare data
        df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
        df.columns = ['label', 'message']
        df = df.drop_duplicates()
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df['cleaned_message'] = df['message'].apply(clean_text)
        
        X = df['cleaned_message']
        y = df['label']
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
        
        # Train Logistic Regression
        model_lr = LogisticRegression(class_weight='balanced', random_state=42)
        model_lr.fit(X_train, y_train)
        
        # Train Naive Bayes
        model_nb = MultinomialNB()
        model_nb.fit(X_train, y_train)
        
        # Save models
        joblib.dump(model_lr, "logistic_model.pkl")
        joblib.dump(model_nb, "naive_bayes_model.pkl")
        joblib.dump(vectorizer, "tfidf.pkl")
        
        st.success("✅ Models trained successfully!")

# Main app interface
st.markdown("---")
st.subheader("🔍 Test the Spam Detector")

# Model selection
model_choice = st.radio(
    "Choose a model:",
    ["Logistic Regression", "Naive Bayes"],
    horizontal=True
)

# Input methods
input_method = st.radio(
    "How would you like to test?",
    ["Single Message", "Multiple Messages", "Sample Messages"],
    horizontal=True
)

if input_method == "Single Message":
    user_message = st.text_area(
        "Enter your message:",
        placeholder="Type or paste your email message here...",
        height=150
    )
    
    if st.button("🔎 Detect Spam", type="primary"):
        if user_message.strip():
            # Clean and vectorize
            cleaned = clean_text(user_message)
            vectorized = vectorizer.transform([cleaned])
            
            # Predict
            model = model_lr if model_choice == "Logistic Regression" else model_nb
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0]
            
            # Display result
            if prediction == 1:
                st.error(f"🚨 **SPAM** (Confidence: {probability[1]:.2%})")
            else:
                st.success(f"✅ **HAM (Not Spam)** (Confidence: {probability[0]:.2%})")
        else:
            st.warning("Please enter a message to analyze.")

elif input_method == "Multiple Messages":
    user_messages = st.text_area(
        "Enter multiple messages (one per line):",
        placeholder="Message 1\nMessage 2\nMessage 3",
        height=200
    )
    
    if st.button("🔎 Detect Spam", type="primary"):
        if user_messages.strip():
            messages = [msg.strip() for msg in user_messages.split('\n') if msg.strip()]
            
            if messages:
                model = model_lr if model_choice == "Logistic Regression" else model_nb
                
                results = []
                for msg in messages:
                    cleaned = clean_text(msg)
                    vectorized = vectorizer.transform([cleaned])
                    prediction = model.predict(vectorized)[0]
                    probability = model.predict_proba(vectorized)[0]
                    
                    results.append({
                        "Message": msg[:50] + "..." if len(msg) > 50 else msg,
                        "Prediction": "SPAM" if prediction == 1 else "HAM",
                        "Confidence": f"{max(probability):.2%}"
                    })
                
                st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("Please enter at least one message.")

else:  # Sample Messages
    sample_messages = [
        "Congratulations! You have won a free prize. Click now to claim it.",
        "Are we still having lunch by noon?",
        "Urgent! Your account has been selected for a cash reward.",
        "Can you send me the meeting notes later today?"
    ]
    
    if st.button("🔎 Test Sample Messages", type="primary"):
        model = model_lr if model_choice == "Logistic Regression" else model_nb
        
        results = []
        for msg in sample_messages:
            cleaned = clean_text(msg)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0]
            
            results.append({
                "Message": msg,
                "Prediction": "SPAM" if prediction == 1 else "HAM",
                "Confidence": f"{max(probability):.2%}"
            })
        
        st.dataframe(pd.DataFrame(results), use_container_width=True)

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Models: Logistic Regression & Naive Bayes")
