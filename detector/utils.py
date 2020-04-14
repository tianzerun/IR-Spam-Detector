import json


def load_json(path):
    with path.open("r") as fp:
        return json.load(fp)


