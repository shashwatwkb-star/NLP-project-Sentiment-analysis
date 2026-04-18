# ==========================================
# SENTIMENT ANALYSIS (IMDB DATASET)
# FINAL CORRECT VERSION
# ==========================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (run once)
nltk.download('stopwords')

# ==========================================
# 2. LOAD DATASET
# ==========================================
df = pd.read_csv("IMDB Dataset.csv")

# Fix column names (your dataset had 0,1)
df.columns = ['review', 'sentiment']

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================================
# 3. TEXT PREPROCESSING
# ==========================================
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]

    return " ".join(words)

df['clean_review'] = df['review'].apply(clean_text)

# ==========================================
# 4. LABELS
# ==========================================
X = df['clean_review']
y = df['sentiment']

# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 6. TF-IDF
# ==========================================
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================================
# 7. MODEL
# ==========================================
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ==========================================
# 8. EVALUATION
# ==========================================
preds = model.predict(X_test_vec)

print("\n===== MODEL EVALUATION =====")
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n")
print(classification_report(y_test, preds))

# ==========================================
# 9. SENTIMENT DISTRIBUTION (FIXED)
# ==========================================
counts = df['sentiment'].value_counts()

plt.figure()
plt.bar(["Negative", "Positive"], counts.values)
plt.title("Sentiment Distribution")
plt.ylabel("Number of Reviews")
plt.xlabel("Sentiment")
plt.show()

# ==========================================
# 10. CUSTOM PREDICTION
# ==========================================
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    return "Positive" if pred == 1 else "Negative"

print("\n===== SAMPLE PREDICTIONS =====")
print("Example 1:", predict_sentiment("This movie was amazing!"))
print("Example 2:", predict_sentiment("Worst movie I have ever seen."))
