import argparse
from pathlib import Path

from elasticsearch import Elasticsearch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from detector.search import EmailIndexSearchClient


def build_feature_matrix(client, features):
    pass


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
    return parser


def main():
    parser = create_args_parser()
    args = parser.parse_args()
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    features = read_spam_words(Path(args.spam_words), stop_words, stemmer)
    client = EmailIndexSearchClient(Elasticsearch("http://localhost:9200"), args.index)
    print(client.get_term_freq("inmail.11", "visit"))


if __name__ == '__main__':
    main()
