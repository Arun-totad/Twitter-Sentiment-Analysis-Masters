#### Twitter Sentimental Analysis Source Code####

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string
import ssl

# Bypass SSL certificate verification for downloading NLTK data
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Handle environments that don't support SSL contexts
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Text Preprocessing Function
def preprocess_text(text):
    """
    Preprocesses the input text by performing the following steps:
    1. Converts text to lowercase.
    2. Removes URLs.
    3. Removes punctuation.
    4. Removes numbers.
    5. Tokenizes the text.
    6. Removes stopwords.
    7. Lemmatizes the words.
    
    Args:
    text (str): The input text to preprocess.
    
    Returns:
    str: The preprocessed text.
    """
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back into a single string
    return ' '.join(words)

# Load and preprocess dataset
df = pd.read_csv('sentiment140.csv', encoding='latin-1', header=None)

# Assign column names to the dataset
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Replace sentiment values (4 to 1 for positive, 0 for negative)
df['sentiment'] = df['sentiment'].replace(4, 1)

# Convert sentiment column to numeric, forcing errors to NaN
df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')

# Drop rows with NaN values in the sentiment column
df = df.dropna(subset=['sentiment'])

# Ensure the sentiment column is of integer type
df['sentiment'] = df['sentiment'].astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression Model
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_vec, y_train)

# Evaluate the model
y_pred_lr = model_lr.predict(X_test_vec)
print(f"\n Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)} \n")
print(f"Classification Report:\n{classification_report(y_test, y_pred_lr)}")

# Predict on new unseen text
new_text = [
    "This place has the worst coffee I have ever tasted!",
    "The concert was unforgettable! I had the best time of my life.",
    "The customer support was very helpful and resolved my issue quickly.",
    "I loved it",
    "This is shit",
    "Not good at all",
    "Could have been better",
    "Whats my name"
]

# Preprocess the new text samples
new_text_preprocessed = [preprocess_text(t) for t in new_text]

# Vectorize the preprocessed text
new_text_vectorized = vectorizer.transform(new_text_preprocessed)

# Predict sentiments using the trained model
predictions = model_lr.predict(new_text_vectorized)

# Print the results with corresponding labels
for i, prediction in enumerate(predictions):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"Text: {new_text[i]}")
    print(f"Predicted Sentiment: {sentiment}")
    print("-" * 50)
    
#User Input
# Input a statement from the user
def user_input():
    new_text = input("Input a statement: ")
    
    # Preprocess the input text
    preprocessed_text = preprocess_text(new_text)
    
    # Vectorize the preprocessed text (wrap it in a list)
    text_vectorized = vectorizer.transform([preprocessed_text])
    
    # Predict the sentiment using the trained model
    prediction = model_lr.predict(text_vectorized)
    
    # Determine the sentiment label
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    # Print the result
    print(f"Text: {new_text}")
    print(f"Predicted Sentiment: {sentiment}")
    print("-" * 50)
    
user_input()

