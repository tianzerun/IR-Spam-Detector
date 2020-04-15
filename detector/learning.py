import argparse
from pathlib import Path
from os import linesep

from elasticsearch import Elasticsearch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

import detector.utils as utils
from detector.search import EmailIndexSearchClient


def train_models(X, y):
    regression_model = LogisticRegression(solver="liblinear", class_weight="balanced")
    bayes_model = MultinomialNB()
    tree_model = DecisionTreeClassifier()

    models = {
        "Logistic Regression": regression_model,
        "Naive Bayes": bayes_model,
        "Decision Tree": tree_model
    }

    for name, model in models.items():
        print(f"Train {name} model")
        model.fit(X, y)

    return models


def test_models(models, email_ids, X, true_labels):
    for name, model in models.items():
        print(f"Test {name} model")
        y_prob = model.predict_proba(X)[:, 1]
        score = roc_auc_score(y_true=true_labels, y_score=y_prob)
        print(f"ROC - area under curve score: {score}{linesep}")

        predictions = sorted([(_id, prob, y_true)
                              for _id, prob, y_true in zip(email_ids, y_prob, true_labels)],
                             key=lambda o: o[1], reverse=True)
        for _id, prob, y_true in predictions[:10]:
            print(f"{_id}: {y_true} ({prob})")


def prepare_learning_data(df):
    email_ids = df[:, 0].toarray().ravel()
    X = df[:, 1:-1]
    y = df[:, -1].toarray().ravel()
    return email_ids, X, y


def build_feature_matrix(client, terms):
    def add_data(row, col, value):
        row_ind.append(row)
        col_ind.append(col)
        data.append(value)

    email_ids = dict()
    num_of_rows = len(client.ids)
    num_of_cols = len(terms) + 2
    id_col = 0
    label_col = len(terms) + 1
    row_ind = []
    col_ind = []
    data = []
    for i, doc_id in enumerate(client.ids):
        email_ids[i] = doc_id
        add_data(i, id_col, i)
        for j, term in enumerate(terms, start=1):
            tf = client.get_term_freq(doc_id, term)
            if tf > 0:
                add_data(i, j, tf)
        label = 1 if client.is_email_spam(doc_id) else 0
        add_data(i, label_col, label)
    return csr_matrix((data, (row_ind, col_ind)), shape=(num_of_rows, num_of_cols)), email_ids


def read_spam_words(path, stop_words, stemmer):
    words = []
    with path.open("r") as fp:
        for line in fp:
            raw = line.strip().lower()
            if raw not in stop_words:
                parts = raw.split()
                if len(parts) > 1:
                    # n-grams
                    continue
                    # words.append(" ".join(stemmer.stem(w) for w in parts))
                else:
                    # uni-gram
                    words.append(stemmer.stem(parts[0]))
    return words


def create_args_parser():
    parser = argparse.ArgumentParser(description="Learn a spam/ham classifier from feature matrices")
    parser.add_argument("index", type=str, action="store",
                        help="the name of a index where emails are stored")
    parser.add_argument("client_pickle", metavar="client-pickle", type=str, action="store",
                        help="the path to an index pickle file")
    parser.add_argument("--spam_words", "-s", metavar="--spam-words", type=str, action="store",
                        help="the path to a file which contains spam words")
    return parser


def train_with_selected_features(spam_words_path):
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    features = read_spam_words(spam_words_path, stop_words, stemmer)
    return features


def convert_email_ids(int_ids, ids_table):
    return [ids_table[_id] for _id in int_ids if _id in ids_table]


def main():
    parser = create_args_parser()
    args = parser.parse_args()
    pickle_file = Path(args.client_pickle)
    out_feature_matrix_file = Path("../data/matrices/unigram.txt")

    if pickle_file.is_file():
        client = utils.load_pickle(pickle_file)
    else:
        client = EmailIndexSearchClient(Elasticsearch("http://localhost:9200"), args.index)

    if args.spam_words is None:
        features = client.get_unigrams()
        print("Features are all uni-grams.")
    else:
        features = train_with_selected_features(Path(args.spam_words))
        print("Features are selected spam words.")

    df, email_ids_table = build_feature_matrix(client, features)
    train, test = train_test_split(df, test_size=0.2)

    email_ids_train, X_train, y_train = prepare_learning_data(train)
    email_ids_test, X_test, y_test = prepare_learning_data(test)
    email_ids_train = convert_email_ids(email_ids_train, email_ids_table)
    email_ids_test = convert_email_ids(email_ids_test, email_ids_table)

    models = train_models(X_train, y_train)
    test_models(models, email_ids_test, X_test, true_labels=y_test)

    dump_svmlight_file(X_train, y_train, f=str(out_feature_matrix_file))

    if not pickle_file.is_file():
        utils.dump_pickle(client, pickle_file)


if __name__ == '__main__':
    main()
