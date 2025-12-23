from flask import Flask, render_template, request
import pickle
import re
import os
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.data.path.append("/opt/render/nltk_data")


app = Flask(__name__)

with open("model/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("nor")
stop_words.discard("never")

def preprocess_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

positive_words = {"good", "great", "excellent", "amazing", "awesome", "nice", "love"}
negative_words = {"bad", "worst", "poor", "terrible", "awful", "hate"}
negation_words = {"not", "no", "never", "hardly"}

os.makedirs("static/charts", exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    review_text = ""
    sentiment = ""
    batch_results = []
    total = pos_count = neg_count = neu_count = 0
    overall = ""
    bar_chart = pie_chart = ""
    keywords = ""
    phrases = ""

    if request.method == "POST":
        action = request.form.get("action")

        if action == "single":
            review = request.form.get("review", "")
            review_text = review
            if review.strip() != "":
                cleaned = preprocess_text(review)
                vector = vectorizer.transform([cleaned])
                probs = model.predict_proba(vector)[0]
                sentiment = classify_sentiment(cleaned, probs)

        elif action == "batch":
            file = request.files.get("file")
            if file and file.filename != "":
                lines = file.read().decode("utf-8").splitlines()
                reviews = [line.strip() for line in lines if line.strip()]

                for idx, review in enumerate(reviews, start=1):
                    cleaned = preprocess_text(review)
                    vector = vectorizer.transform([cleaned])
                    probs = model.predict_proba(vector)[0]
                    s = classify_sentiment(cleaned, probs)
                    batch_results.append(f"Review {idx} → {s}")

                    total += 1
                    if "Positive" in s:
                        pos_count += 1
                    elif "Negative" in s:
                        neg_count += 1
                    else:
                        neu_count += 1

                overall = max(
                    [("Positive", pos_count), ("Neutral", neu_count), ("Negative", neg_count)],
                    key=lambda x: x[1]
                )[0]

                labels = ["Positive", "Neutral", "Negative"]
                values = [pos_count, neu_count, neg_count]

                plt.figure()
                plt.bar(labels, values, color=["#4CAF50", "#FFC107", "#F44336"])
                bar_path = "static/charts/bar.png"
                plt.savefig(bar_path)
                plt.close()

                plt.figure()
                plt.pie(values, labels=labels, autopct="%1.1f%%", colors=["#4CAF50", "#FFC107", "#F44336"], startangle=140, wedgeprops={'edgecolor': 'white'})
                pie_path = "static/charts/pie.png"
                plt.savefig(pie_path)
                plt.close()

                bar_chart = bar_path
                pie_chart = pie_path

                all_tokens = []
                for review in reviews:
                    cleaned = preprocess_text(review)
                    all_tokens.extend(cleaned.split())

                keyword_counts = Counter(all_tokens)
                keywords = ", ".join([w for w, _ in keyword_counts.most_common(10)])

                bigrams = []
                for i in range(len(all_tokens) - 1):
                    bigrams.append(all_tokens[i] + " " + all_tokens[i + 1])

                phrase_counts = Counter(bigrams)
                phrases = ", ".join([p for p, _ in phrase_counts.most_common(10)])

    return render_template(
        "index.html",
        sentiment=sentiment,
        batch_results=batch_results,
        total=total,
        pos_count=pos_count,
        neg_count=neg_count,
        neu_count=neu_count,
        overall=overall,
        bar_chart=bar_chart,
        pie_chart=pie_chart,
        keywords=keywords,
        phrases=phrases,
        review_text=review_text
    )

def classify_sentiment(cleaned_review, probs):
    tokens = cleaned_review.split()
    neg_prob, pos_prob = probs[0], probs[1]

    for i in range(len(tokens) - 1):
        if tokens[i] in negation_words and tokens[i + 1] in positive_words:
            return "Negative 😞"
        if tokens[i] in negation_words and tokens[i + 1] in negative_words:
            return "Positive 😊"

    if any(word in positive_words for word in tokens):
        return "Positive 😊"
    if any(word in negative_words for word in tokens):
        return "Negative 😞"

    if pos_prob >= 0.55:
        return "Positive 😊"
    if neg_prob >= 0.55:
        return "Negative 😞"

    return "Neutral 😐"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

