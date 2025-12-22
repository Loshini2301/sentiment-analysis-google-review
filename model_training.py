import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

data = pd.read_csv("dataset/Restaurant_Reviews.tsv", delimiter="\t")

reviews = data["Review"]
labels = data["Liked"]

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

cleaned_reviews = []

for review in reviews:
    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower()
    words = review.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    review = " ".join(words)
    cleaned_reviews.append(review)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_reviews)
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


import pickle

with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
