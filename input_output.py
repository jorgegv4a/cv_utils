import json
import pickle


def save_json(filepath, data, indent=4):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def save_pickle(filepath, data):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)