import json
import pickle
import datetime


class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return json.JSONEncoder.default(obj)


class _JSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        ret = {}
        for key, value in obj.items():
            if isinstance(value, str):
                try:
                    ret[key] = datetime.datetime.fromisoformat(value)
                except ValueError:
                    pass
            else:
                ret[key] = value
        return ret


def save_json(filepath, data, indent=4):
    with open(filepath, "w") as f:
        # json.dump(data, f, indent=indent)
        f.write(json.dumps(data, cls=_JSONEncoder, indent=4))


def load_json(filepath):
    with open(filepath, "r") as f:
        # return json.load(f)
        return json.loads(f.read(), cls=_JSONDecoder)


def save_pickle(filepath, data):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)