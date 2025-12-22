from flask import Flask, render_template, request
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)

with open("model/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = ""
    if request.method == "POST":
        review = request.form["review"]
        cleaned_review = preprocess_text(review)
        vector = vectorizer.transform([cleaned_review])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            sentiment = "Positive 😊"
        else:
            sentiment = "Negative 😞"

    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

