import re
from glob import glob

import pandas as pd


def extract_kv_pairs(entry):
    for line in entry.split("\n"):
        key_values = re.split("\s{2}|:", line)
        key = key_values[0]

        try:
            # all elements we are interested in are floats, therefore we dont need to join anything
            value = float("".join(key_values[1:]))
            yield (key_mapping[key], value)
        except:
            pass


if __name__ == '__main__':
    for file in glob("* accuracies.txt"):
        try:
            docname = file.split(" ")[0]
            with open(file) as f:
                entries = f.read().split("\n\n")[1:]

            key_mapping = {'curr max dist': "dist", 'unk': "unk", 'known': "known", 'total': "total"}

            df = pd.DataFrame([dict(list(extract_kv_pairs(entry))) for entry in entries]).set_index("dist")
            df.plot(title=docname).get_figure().savefig(f"{docname}.png")
        except:
            print("a error while parsing", file, "occured")
