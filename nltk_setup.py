import nltk
import os

nltk_data_dir = "/opt/render/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)

nltk.data.path.append(nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir, quiet=True)


