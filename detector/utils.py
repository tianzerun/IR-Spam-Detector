import json
import pickle


def load_json(path):
    with path.open("r") as fp:
        return json.load(fp)


def dump_pickle(obj, path):
    with path.open("wb") as fp:
        pickle.dump(obj, fp)


def load_pickle(path):
    with path.open("rb") as fp:
        return pickle.load(fp)
