import re
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

reviews = []
labels = []

with open("dataset/reviews.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if line.startswith("__label__2"):
            labels.append(1)
            review = line.replace("__label__2", "").strip()
            reviews.append(review)

        elif line.startswith("__label__1"):
            labels.append(0)
            review = line.replace("__label__1", "").strip()
            reviews.append(review)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("nor")
stop_words.discard("never")


cleaned_reviews = []

for review in reviews:
    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower()
    words = review.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    review = " ".join(words)
    cleaned_reviews.append(review)

vectorizer = TfidfVectorizer(ngram_range=(1,2))
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

with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
