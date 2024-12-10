# Twitter Sentiment Analysis

This project performs sentiment analysis on Twitter data using a Logistic Regression model. The analysis involves preprocessing text data, vectorizing it using TF-IDF, and training a Logistic Regression model to classify sentiments as positive or negative.

## Table of Contents
- Installation
- Usage
- Project Structure
- Data
- Model
- Evaluation
- Prediction


## Installation

Clone the repository:
   ```bash
   git clone https://github.com/Arun-totad/Twitter-Sentiment-Analysis-Masters.git
   cd twitter-sentiment-analysis
   ```
Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn nltk
   ```
Download NLTK data:
   ```bash
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

## Usage
Preprocess the text data: The preprocess_text function performs the following steps:
* Converts text to lowercase
* Removes URLs, punctuation, and numbers
* Tokenizes the text
* Removes stopwords
* Lemmatizes the words

* Load and preprocess the dataset:
```
df = pd.read_csv('sentiment140.csv', encoding='latin-1', header=None)
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
df['sentiment'] = df['sentiment'].replace(4, 1)
df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
df = df.dropna(subset=['sentiment'])
df['sentiment'] = df['sentiment'].astype(int)
```

Split the data into training and testing sets:
```
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)
```

Vectorize the text data using TF-IDF:
```
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```

Train the Logistic Regression model:
```
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_vec, y_train)
```

Evaluate the model:
```
y_pred_lr = model_lr.predict(X_test_vec)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_lr)}")
```

Predict sentiments on new text:
```
new_text = ["This place has the worst coffee I have ever tasted!", "The concert was unforgettable! I had the best time of my life."]
new_text_preprocessed = [preprocess_text(t) for t in new_text]
new_text_vectorized = vectorizer.transform(new_text_preprocessed)
predictions = model_lr.predict(new_text_vectorized)
for i, prediction in enumerate(predictions):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"Text: {new_text[i]}")
    print(f"Predicted Sentiment: {sentiment}")
```

User input for sentiment prediction:
```
def user_input():
    new_text = input("Input a statement: ")
    preprocessed_text = preprocess_text(new_text)
    text_vectorized = vectorizer.transform([preprocessed_text])
    prediction = model_lr.predict(text_vectorized)
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"Text: {new_text}")
    print(f"Predicted Sentiment: {sentiment}")
user_input()
```

## Project Structure
sentiment140.csv: The dataset containing Twitter data.

twitter_sentiment_analysis.py: The main script for preprocessing, training, evaluating, and predicting sentiments.

## Data
The dataset used is the Sentiment140 dataset, which contains 1.6 million tweets labeled as positive (1) or negative (0).

## Model
The model used is a Logistic Regression classifier trained on TF-IDF vectorized text data.

## Evaluation
The model is evaluated using accuracy and classification report metrics.

## Prediction
The model can predict the sentiment of new, unseen text data.
