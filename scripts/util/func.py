import json


def param_to_key(param):
    def __serialize(x):
        try:
            return dict.fromkeys(x)
        except:
            return str(x)

    return json.dumps(dict([(k, v) for k, v in param.items() if k not in {'memlimit'}]),
                      default=__serialize)
