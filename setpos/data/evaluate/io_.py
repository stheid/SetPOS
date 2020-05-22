from os import path

import pandas as pd
import csv, re
from bidict import bidict
import numpy as np
from configparser import ConfigParser
from os import path

config = ConfigParser()
project_root = path.join(path.dirname(__file__), "../../../")
config.read(path.join(project_root, "path.conf"))
ipath = path.join(project_root, config["DEFAULT"]["datapath"])


def load(dir=ipath):
    df = pd.read_csv(path.join(dir, "5_corenlp.csv"), sep=';', quoting=csv.QUOTE_ALL, header=None, na_filter=False)
    df.columns = "word,sentID,isunk,target,pred,posterior,constrainedtags".split(",")
    df.posterior = df.posterior.apply(lambda x: np.array(eval(x)))
    df.constrainedtags = df.constrainedtags.apply(lambda x: tuple(re.match('\[(.*)\]', x)[1].split(', ')))

    with open(path.join(dir, "5_tags.csv")) as f:
        tags = f.readline().strip().split(", ")
    tagsdict = bidict(enumerate(tags))
    return df, tagsdict
