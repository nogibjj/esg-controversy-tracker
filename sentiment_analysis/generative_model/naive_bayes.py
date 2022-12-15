import sys
from datetime import datetime as dt
import random

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

sys.path.append("../..")
from pre_processing_scripts import pre_processing

random_state = 12321
dataset_path = "/workspaces/esg-controversy-tracker/dataset/news_sentiment.csv"

# Limit the dataset to n rows
data = pd.read_csv(dataset_path)
data["confidence"] = data["confidence"].abs()
data = data[data["confidence"] >= 0.99]

max_class_samples = 25000
data = data.sample(frac=1, random_state=random_state).reset_index()
pos_sample = data[data["sentiment"] == "POSITIVE"][0:max_class_samples]
neg_sample = data[data["sentiment"] == "NEGATIVE"][0:max_class_samples]
data = pd.concat([pos_sample, neg_sample])

data["sentiment"] = [0 if x == "NEGATIVE" else 1 for x in data["sentiment"]]


def data_preprocess(data):
    """
    Wrapper Function to perform pre-processing and vectorization of the data
    Main functions are written in pre_processing_scripts/pre_processing.py
    """
    # Pre-process Data
    data = data.sample(frac=1, random_state=random_state).reset_index()
    data["Title"] = pre_processing.remove_stopwords(data["Title"])
    data["Title"] = pre_processing.remove_non_alphanumeric(data["Title"])
    data["Title"] = np.vectorize(pre_processing.stem_text)(data["Title"])
    data["Title"] = pre_processing.remove_numbers(data["Title"])

    # Split the data into training and testing data
    title = data["Title"].values
    count_vec = CountVectorizer()
    title = count_vec.fit_transform(title).toarray()
    vocabulary = count_vec.get_feature_names_out()

    labels = data["sentiment"].values
    train_x, test_x, train_y, test_y = train_test_split(
        title, labels, stratify=labels, random_state=random_state
    )
    return train_x, test_x, train_y, test_y, vocabulary


def train_model(train_x, test_x, train_y, test_y, model_type=MultinomialNB()):
    """
    Function to train the model
    """
    naive_bayes_classifier = model_type
    naive_bayes_model = naive_bayes_classifier.fit(train_x, train_y)
    pred_y = naive_bayes_model.predict(test_x)
    print(accuracy_score(pred_y, test_y))
    return naive_bayes_model


def generate_synthetic_data(naive_bayes_model, vocabulary):

    total_samples_required = len(data)
    prior_word_prob = np.exp(naive_bayes_model.feature_log_prob_)

    # Generating Positive Samples - class 1
    pos_sentences = []
    for n in range((total_samples_required // 2)):
        word_list = random.choices(
            vocabulary, prior_word_prob[1], k=random.randint(5, 15)
        )
        pos_sentences.append(" ".join(word_list))
    pos_df = pd.DataFrame({"Title": pos_sentences, "sentiment": 1})

    # Generating Negative Samples - class 0
    neg_sentences = []
    for n in range((total_samples_required // 2)):
        word_list = random.choices(
            vocabulary, prior_word_prob[0], k=random.randint(5, 15)
        )
        neg_sentences.append(" ".join(word_list))
    neg_df = pd.DataFrame({"Title": neg_sentences, "sentiment": 0})
    synthetic_data = pd.concat([pos_df, neg_df])
    synthetic_data.to_csv(
        f'synthetic_data_{dt.now().strftime("%m_%d_%Y_%H_%M_%S")}.csv'
    )

    return synthetic_data


train_x, test_x, train_y, test_y, vocabulary = data_preprocess(data)
model = train_model(train_x, test_x, train_y, test_y)

#synthetic_data = generate_synthetic_data(model, vocabulary)
synthetic_dataset_path = "/workspaces/esg-controversy-tracker/sentiment_analysis/generative_model/synthetic_data_12_13_2022_23_34_50.csv"
synthetic_data = pd.read_csv(synthetic_dataset_path)
train_x, test_x, train_y, test_y, vocabulary = data_preprocess(synthetic_data)
model = train_model(train_x, test_x, train_y, test_y)
