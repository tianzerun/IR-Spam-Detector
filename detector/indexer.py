import argparse
from pathlib import Path

from elasticsearch import Elasticsearch

import detector.utils as utils


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
