import json
import re
from configparser import ConfigParser
from os import path

config = ConfigParser()
project_root = path.join(path.dirname(__file__), "../../../")
config.read(path.join(project_root, "path.conf"))
ifile = path.join(project_root, config["DEFAULT"]["datapath"], config["DEFAULT"]["intergram_export_file"])
ofile = path.join(project_root, config["DEFAULT"]["datapath"], config["DEFAULT"]["sentsplit_export_file"])


def load(file=ifile):
    # works on results from XMLtext Exports from the InterGramm Java Project

    with open(file) as f:
        docs = dict([re.match("\[(\w+)\](.*)", line).groups() for line in f])

    return docs


def dump(docs, file=ofile):
    with open(file, "w") as f:
        json.dump(docs, f)
