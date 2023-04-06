import pandas as pd
import numpy as np
import pickle
import functions.text_preprocessing as tp


def load_data():
    # load data
    reddit_content = pd.read_csv("data_store/scraped/reddit_store.csv")

    return reddit_content

def vectorize_data(reddit_content):
    # load vectorizer
    vectorizer = pickle.load(open("models/vectorizer.pickle", "rb"))

    # vectorize text
    vectorized_text = vectorizer.transform(reddit_content["text"])

    return vectorized_text

def predict_targets(vectorized_text):
    # load model
    model = pickle.load(open("models/PassiveAggressiveClassifier_model.pkl", "rb"))

    # predict targets
    predictions = model.predict(vectorized_text)

    return predictions