import argparse
from pathlib import Path

from elasticsearch import Elasticsearch
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import detector.utils as utils
from detector.search import EmailIndexSearchClient


def train_model(X, y):
    model = LogisticRegression(solver="lbfgs", class_weight="balanced")
    model.fit(X, y)
    return model


def test_model(model, X, true_labels):
    y_prob = model.predict_proba(X)[:, 1]
    score = roc_auc_score(y_true=true_labels, y_score=y_prob)
    print(score)


def prepare_learning_data(df):
    email_ids = df.index.values
    X = df.values[:, :-1]
    y = df.values[:, -1].astype(np.uint8)
    return email_ids, X, y


def build_feature_matrix(client, terms):
    key = "EID"
    titles = [key] + terms + ["LABEL"]
    content = []
    for doc_id in client.ids:
        row = [doc_id]
        for term in terms:
            row.append(client.get_term_freq(doc_id, term))
        label = 1 if client.is_email_spam(doc_id) else 0
        row.append(label)
        content.append(row)
    return pd.DataFrame(content, columns=titles).set_index(key)


def read_spam_words(path, stop_words, stemmer):
    words = []
    with path.open("r") as fp:
        for line in fp:
            raw = line.strip().lower()
            if raw not in stop_words:
                parts = raw.split()
                if len(parts) > 1:
                    # n-grams
                    words.append(" ".join(stemmer.stem(w) for w in parts))
                else:
                    # uni-gram
                    words.append(stemmer.stem(parts[0]))
    return words


def create_args_parser():
    parser = argparse.ArgumentParser(description="Learn a spam/ham classifier from feature matrices")
    parser.add_argument("spam_words", metavar="spam-words", type=str, action="store",
                        help="the path to a file which contains spam words")
    parser.add_argument("index", type=str, action="store",
                        help="the name of a index where emails are stored")
    parser.add_argument("client_pickle", metavar="client-pickle", type=str, action="store",
                        help="the path to an index pickle file")
    return parser


def main():
    parser = create_args_parser()
    args = parser.parse_args()
    pickle_file = Path(args.client_pickle)

    if pickle_file.is_file():
        client = utils.load_pickle(pickle_file)
    else:
        client = EmailIndexSearchClient(Elasticsearch("http://localhost:9200"), args.index)
    print("Finish building client")

    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    features = read_spam_words(Path(args.spam_words), stop_words, stemmer)

    df = build_feature_matrix(client, features)
    train, test = train_test_split(df, test_size=0.2)
    print("Finish building feature matrix")

    email_ids_train, X_train, y_train = prepare_learning_data(train)
    email_ids_test, X_test, y_test = prepare_learning_data(test)
    model = train_model(X_train, y_train)
    test_model(model, X_test, true_labels=y_test)

    if not pickle_file.is_file():
        utils.dump_pickle(client, pickle_file)


if __name__ == '__main__':
    main()
