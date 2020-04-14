import argparse
import email
import os
import re
from pathlib import Path
from enum import Enum
from os import linesep

from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch, helpers
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

BASIC_TOKEN_REGEX = re.compile(r"[\w]*[a-zA-Z]+[\w]")


class SpamLabel(Enum):
    SPAM = "spam"
    HAM = "ham"
    UNDECIDED = "undecided"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class DataCategory(Enum):
    TRAIN = "train"
    TEST = "test"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class Email(object):
    def __init__(self, subject, raw_body, exclude_terms=None, stemmer=PorterStemmer()):
        if exclude_terms is None:
            self._exclude_terms = set(stopwords.words("english"))
        self.subject = subject
        self.raw_body = raw_body
        self.cleaned_subject = clean_text(self.subject, exclude_terms, stemmer)
        self.cleaned_body = clean_text(self.raw_body, exclude_terms, stemmer)
        self.label = SpamLabel.UNDECIDED

    def add_label(self, label):
        self.label = label


def get_emails_labels(path):
    labels = dict()
    with path.open("r") as fp:
        for line in fp:
            label, file_path = line.split(" ")
            filename = Path(file_path.strip()).name
            labels[filename] = label.lower()
    return labels


def email_file_reader(path):
    with path.open("rb") as fp:
        return fp.read()


def clean_text(text, exclude_terms=None, stemmer=None):
    def no_stem(w):
        return w

    def stem(w):
        return stemmer.stem(w)

    stemmer_handler = no_stem
    if exclude_terms is None:
        exclude_terms = set()
    if stemmer is not None:
        stemmer_handler = stem

    cleaned = " ".join(text.split())
    tokens = []
    for term in re.findall(BASIC_TOKEN_REGEX, cleaned):
        term = term.lower()
        if term not in exclude_terms:
            term = stemmer_handler(term)
            tokens.append(term)
    return " ".join(tokens)


def email_builder(path):
    def collect_html_text(html):
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text().strip()
        if len(text) == 0:
            text = html
        return text

    def email_part_handler(part):
        no_text = ""
        content_type = part.get_content_type()
        content_disposition = part.get_content_disposition()
        if content_disposition != "attachment":
            if content_type == "text/plain" or content_type == "text/plain charset=us-ascii":
                return part.get_payload()
            elif content_type == "text/html":
                return collect_html_text(part.get_payload())
            else:
                return no_text
        else:
            return no_text

    raw = email_file_reader(path)
    msg = email.message_from_bytes(raw)
    subject = str(msg.get("subject", ""))
    if msg.is_multipart():
        body = " ".join([email_part_handler(p) for p in msg.walk()])
    else:
        body = email_part_handler(msg)

    return Email(subject, body)


def read_emails(folder, file_prefix="inmail."):
    files = filter(lambda f: f.startswith(file_prefix), os.listdir(folder))
    files = sorted(files, key=lambda f: int(f[len(file_prefix):]))

    for filename in files:
        file_path = folder.joinpath(filename)
        yield email_builder(file_path), filename


def prepare_index_requests(email_generator, labels, index, train=0.8):
    counter = 0
    for mail, filename in email_generator:
        body = {
            "_index": index,
            "_id": filename,
            "subject": mail.cleaned_subject,
            "content": mail.cleaned_body,
            "label": str(labels.get(filename, SpamLabel.UNDECIDED)),
            "split": "train"
        }
        yield body
        counter += 1
        if counter > 10:
            break


def create_args_parser():
    parser = argparse.ArgumentParser(description="Preprocess email data.")
    parser.add_argument("in_folder", metavar="in-folder", type=str, action="store",
                        help="the path to a folder where raw email data is stored")
    parser.add_argument("labels", type=str, action="store",
                        help="the path to a file which contains spam/ham labels of emails")
    parser.add_argument("index", type=str, action="store",
                        help="the name of a index where emails will be stored")
    return parser


def main():
    args_parser = create_args_parser()
    args = args_parser.parse_args()
    emails_loc = Path(args.in_folder)
    labels_loc = Path(args.labels)
    labels = get_emails_labels(labels_loc)
    client = Elasticsearch("http://localhost:9200")
    helpers.bulk(client=client, actions=prepare_index_requests(read_emails(emails_loc), labels, args.index))


if __name__ == "__main__":
    main()
    # print(email_builder(Path("/Users/tianzerun/Desktop/CS6200/IR_data/trec07p/data/inmail.8")))
