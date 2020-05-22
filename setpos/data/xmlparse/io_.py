from configparser import ConfigParser

from os import path
from zipfile import ZipFile

config = ConfigParser()
project_root = path.join(path.dirname(__file__), "../../../")
config.read(path.join(project_root, "path.conf"))
ifile = path.join(project_root, config["DEFAULT"]["datapath"], config["DEFAULT"]["raw_data_file"])
ofile = path.join(project_root, config["DEFAULT"]["datapath"], config["DEFAULT"]["intergram_export_file"])


def load(file=ifile):
    # load zipfile
    with ZipFile(file) as z:
        for f in z.namelist():
            with z.open(f) as f:
                # yield list of strings where each string contains the contents of one zipfile
                yield f.read()


def dump(docs, file=ofile):
    with open(file, "w") as f:
        for k, v in docs.items():
            f.write(f'[{k}]{v}\n')
