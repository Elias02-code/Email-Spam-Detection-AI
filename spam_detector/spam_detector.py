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

# Split dataset into training and testing sets
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,                          # 80% data for training and 20% for testing
    random_state=42,                        # ensures reproducibility
    stratify=y                              # important for imbalanced datasets

    )

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,                       # Limit vocabulary size, keeps only the top 5000 most informative words/phrases
    ngram_range=(1, 2),                      # Include unigram and bigrams too (e.g., "free entry")
    min_df=2,                                # Ignore terms that appear in fewer than 2 messages
    max_df=0.95                              # Ignore terms in >95% of documents, such words are too common to be useful
)

# Fit and transform train data
X_train = vectorizer.fit_transform(X_train_text)

# Only transform test data
X_test = vectorizer.transform(X_test_text)

# Confirm vectorizer worked
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# To also confirm the vectorizer worked
print(len(vectorizer.vocabulary_))
print(list(vectorizer.vocabulary_.keys())[:20])

# Confirm numeric features were generated
print(X_train[0])

# Defining the weight model(using Logistic Regression)
model_weighted = LogisticRegression(class_weight='balanced', random_state=42)
print(model_weighted.fit(X_train, y_train))

# Defining the y_pred(weighted)
y_pred_weighted = model_weighted.predict(X_test)

# Analysing the distribution of predicted labels
y_pred_series = pd.Series(y_pred_weighted)
print("Distribution of predicted labels:")
print(y_pred_series.value_counts())

# Analysing the distribution of actual labels in y_test:
print("Distribution of actual labels in y_test:")
print(y_test.value_counts())

# Calculating the overall correctness of the model
accuracy = accuracy_score(y_test, y_pred_weighted)

# Calculating how many predicted spam emails were actually spam
precision = precision_score(y_test, y_pred_weighted)

# Calculating how many real spam emails were correctly detected
recall = recall_score(y_test, y_pred_weighted)

# Balance between precision and recall
f1 = f1_score(y_test, y_pred_weighted)

print("Logistic Regression Evaluation Metrics:")

# evaluating the prediction using Accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred_weighted))
# evaluating the prediction using Precision_score
print("Precision:", precision_score(y_test, y_pred_weighted))
# evaluating the prediction using Recall_score
print("Recall:", recall_score(y_test, y_pred_weighted))
# evaluating the prediction using F1_score
print("F1_score:", f1_score(y_test, y_pred_weighted))

# evaluating the prediction using confusion_matrix
cm = confusion_matrix(y_test, y_pred_weighted)
print("Confusion Matrix:")
print(cm)

# Interpreting the Confusion Matrix

# The confusion matrix provides a detailed breakdown of the model's performance:
  #True Negatives (TN): The number of 'ham' messages correctly identified as 'ham'.
  #False Positives (FP): The number of 'ham' messages incorrectly identified as 'spam' (Type I error).
  #False Negatives (FN): The number of 'spam' messages incorrectly identified as 'ham' (Type II error).
  #True Positives (TP): The number of 'spam' messages correctly identified as 'spam'.

# For a 2x2 matrix, it's typically structured as:

#[[TN, FP],
 #[FN, TP]]

 #This matrix is structured as follows, where 0 represents 'ham' and 1 represents 'spam':

    #The rows represent the actual classes.
    #The columns represent the predicted classes.

#Here's what each number means:

    #True Negatives (TN): 890
        #These are the 890 'ham' messages that were actually ham and were correctly predicted as ham by the model.

    #False Positives (FP): 13
        #These are the 13 'ham' messages that were actually ham but were incorrectly predicted as spam by the model. This is a Type I error.

    #False Negatives (FN): 9
        #These are the 9 'spam' messages that were actually spam but were incorrectly predicted as ham by the model. This is a Type II error, meaning the model missed these spam messages.

    #True Positives (TP): 122
        #These are the 122 'spam' messages that were actually spam and were correctly predicted as spam by the model.

#In summary:

    #The top-left 890 means 888 actual ham messages were correctly identified as ham.
    #The top-right 13 means 13 actual ham messages were wrongly identified as spam.
    #The bottom-left 9 means 9 actual spam messages were wrongly identified as ham.
    #The bottom-right 122 means 122 actual spam messages were correctly identified as spam.

# Displaying detailed classification report
print("Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_weighted))

# Store Logistic Regression results for comparison with Naive Bayes
logistic_regression_results = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
}

print(logistic_regression_results)

# Testing the Logistic Regression model on a few custom messages
sample_messages = [
    "Congratulations! You have won a free prize. Click now to claim it.",
    "Are we still having lunch by noon?",
    "Urgent! Your account has been selected for a cash reward.",
    "Can you send me the meeting notes later today?"
]

#Transformed the sample messages using the same vectorizer that was fitted on the training data
sample_messages_vectorized = vectorizer.transform(sample_messages)

# Predict whether each message is spam or ham using Logistic Regression
sample_predictions_weighted = model_weighted.predict(sample_messages_vectorized)

for message, prediction in zip(sample_messages, sample_predictions_weighted):
    label = "Spam" if prediction == 1 else "Ham"
    print(f"Message: \"{message}\"")
    print(f"Predicted Label: {label}\n")

print(joblib.dump(model_weighted, "logistic_model.pkl"))
print(joblib.dump(vectorizer, "tfidf.pkl"))

# Initialize the Multinomial Naive Bayes model, which is suitable for text-based classification tasks such as email spam detection where features represent word frequencies.
nb_model = MultinomialNB()

# Training the Naive Bayes model using the training dataset.
print(nb_model.fit(X_train, y_train))

# Generating predictions on the test dataset.
y_pred_nb = nb_model.predict(X_test)

# Converting predictions to a pandas Series to examine the distribution of predicted labels. This helps identify whether the model is biased toward predicting spam or non-spam emails.
y_pred_nb_series = pd.Series(y_pred_nb)
print("Distribution of predicted labels (Naive Bayes):")
print(y_pred_nb_series.value_counts())

# Generating the confusion matrix to evaluate the classification performance of the Naive Bayes model.
cm_nb = confusion_matrix(y_test, y_pred_nb)
print("Confusion Matrix (Naive Bayes):")
print(cm_nb)

# Calculating overall correctness of the model
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Calculating how many predicted spam emails were actually spam
precision_nb = precision_score(y_test, y_pred_nb)

# Calculating how many real spam emails were correctly detected
recall_nb = recall_score(y_test, y_pred_nb)

# Calculating the b alance between precision and recall
f1_nb = f1_score(y_test, y_pred_nb)

print("Naive Bayes Evaluation Metrics:")
print(f"Accuracy: {accuracy_nb:.4f}")
print(f"Precision: {precision_nb:.4f}")
print(f"Recall: {recall_nb:.4f}")
print(f"F1 Score: {f1_nb:.4f}")

# Storing Naive Bayes evaluation results for comparison
naive_bayes_results = {
    "Accuracy": accuracy_nb,
    "Precision": precision_nb,
    "Recall": recall_nb,
    "F1 Score": f1_nb
}
print(naive_bayes_results)

# Display classification report for Naive Bayes
print("Classification Report (Naive Bayes):")
print(classification_report(y_test, y_pred_nb))

# Testing the Naive Bayes model on a few custom messages. This allows us to see how the model classifies real-world examples outside of the original dataset.
sample_messages = [
    "Congratulations! You have won a free prize. Click now to claim it.",
    "Are we still having lunch by noon?",
    "Urgent! Your account has been selected for a cash reward.",
    "Can you send me the meeting notes later today?"
]

# Transforming the sample messages using the same vectorizer that was fitted on the training data
sample_messages_vectorized = vectorizer.transform(sample_messages)

# Predicting whether each message is spam or ham
sample_predictions = nb_model.predict(sample_messages_vectorized)

# Displaying the results
for message, prediction in zip(sample_messages, sample_predictions):
    label = "Spam" if prediction == 1 else "Ham"
    print(f"Message: \"{message}\"")
    print(f"Predicted Label: {label}\n")

    # Creating a comparison table between Logistic Regression and Naive Bayes
comparison_df = pd.DataFrame({
    "Logistic Regression": logistic_regression_results,
    "Naive Bayes": naive_bayes_results
})
print(comparison_df)

print(joblib.dump(nb_model, "naive_bayes_model.pkl"))
