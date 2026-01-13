# Import libraries

import streamlit as st 

st.title("📧 Email Spam Detector")
st.write("This app uses Machine Learning to identify spam messages.")
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the SMS spam dataset with proper text encoding
df = pd.read_csv("Streamlit_app/spam.csv", encoding='latin1')
st.subheader("Dataset Preview")
st.dataframe(df.head())
print(df.shape)
print(df.columns)

# Clean unnecessary columns and rename the remaining columns
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Display column names to confirm relevant features
print(df.columns)
print(df.shape)

# Check the number of duplicate rows in the dataset
print(df.duplicated().sum())

# Remove duplicate messages to avoid model bias
df = df.drop_duplicates()

# Confirm removed duplicates
print(df.shape)

# Check for missing rows
print(df.isnull().sum())

# Display the count of spam and ham messages after cleaning
print(df['label'].value_counts())

# Create a new feature representing message length
df['message_length'] = df['message'].apply(len)
print(df['message_length'])
print(df.columns)

# Generate descriptive statistics of message length by class
print(df.groupby('label')['message_length'].describe())

# Map labels to numeric (ham: 0, spam: 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Confirm mapped feature
print(df['label'])

# Download NLTK data... Commented due to network instabilities at the time.
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Text Preprocessing Function... The newly implemented format using ENGLISH_STOP_WORDS works fine as well.
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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
# Replace money symbols (spam often has $ or £)
    text = re.sub(r'£|\$', ' money ', text)
# Replace phone numbers
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', ' phonenumber ', text)
# Replace numbers (but keep as token - spam uses numbers a lot)
    text = re.sub(r'\d+', ' number ', text)
# Remove punctuation but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
# Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
# Tokenize, remove stopwords, and stem
    words = text.split()

    return ' '.join(words)

# Apply cleaning
df['cleaned_message'] = df['message'].apply(clean_text)
print(df.head())

# Select input and output features
X = df['cleaned_message']
y = df['label']
print(df.columns)

# Checking if the text cleaning worked
print("\nExample cleaning:")
print("Original:", df['message'].iloc[0])
print("Cleaned :", df['cleaned_message'].iloc[0])
