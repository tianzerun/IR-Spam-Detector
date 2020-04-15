import argparse
from pathlib import Path

from elasticsearch import Elasticsearch, helpers

import detector.utils as utils
from detector.preprocess import SpamLabel


class EmailIndexSearchClient(object):
    def __init__(self, client, index):
        self._index = index
        self._labels = self._get_labels(client, index)
        self._term_vectors = self._get_term_vectors(client, index)

    @property
    def ids(self):
        return list(self._labels.keys())

    @classmethod
    def _get_labels(cls, client, index):
        f_id = "_id"
        f_label = "label"
        body = {
            "query": {
                "match_all": {}
            },
            "stored_fields": [f_id, f_label]
        }
        return {item[f_id]: SpamLabel.from_str(item["fields"][f_label][0])
                for item in helpers.scan(client, index=index, query=body)}

    def _get_term_vectors(self, client, index):
        ids = list(self._labels.keys())
        term_vectors = dict()
        size = 50
        p = 0
        while p < len(ids):
            res = client.mtermvectors(
                index=index,
                fields=["content"],
                ids=ids[p:p + size],
                term_statistics=False,
                field_statistics=False,
                offsets=False,
                positions=False
            )
            for doc in res["docs"]:
                freq_by_term = dict()
                if "content" in doc["term_vectors"]:
                    for term, value in doc["term_vectors"]["content"]["terms"].items():
                        freq_by_term[term] = value["term_freq"]
                    term_vectors[doc["_id"]] = freq_by_term
            p += size
        return term_vectors

    def get_term_freq(self, doc_id, term):
        return 0 if doc_id not in self._term_vectors else self._term_vectors[doc_id].get(term, 0)

    def is_email_spam(self, doc_id):
        return self._labels.get(doc_id, SpamLabel.UNDECIDED) is SpamLabel.SPAM


def create_index(client, index, index_config):
    try:
        if not client.indices.exists(index):
            client.indices.create(index=index, body=index_config)
            print(f"Created index {index}.")
    except Exception as ex:
        raise ex


def create_args_parser():
    parser = argparse.ArgumentParser(description="Create an index in ElasticSearch.")
    parser.add_argument("index", type=str, action="store",
                        help="the name of index to be created")
    parser.add_argument("configs", type=str, action="store",
                        help="the path to the index configuration file")
    return parser


def main():
    parser = create_args_parser()
    args = parser.parse_args()
    configs = utils.load_json(Path(args.configs))
    client = Elasticsearch("http://localhost:9200")
    create_index(client, args.index, configs)


if __name__ == '__main__':
    main()
