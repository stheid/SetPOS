import json
import re
from collections import ChainMap
from configparser import ConfigParser
from glob import glob
from multiprocessing.pool import Pool
from os import path
from os.path import basename

import pandas as pd


def get_files_for_suffix(suffix):
    return [file for file in glob(f"out/*{suffix}") if re.fullmatch(r'\w{,6}' + suffix, basename(file))]


def get_values(suffix):
    dfs = [pd.read_csv(file) for file in get_files_for_suffix(suffix)]
    df = pd.DataFrame(
        [(df_tmp.total.max(), df_tmp.total.iloc[-1], len(eval(df_tmp.train_docs.iloc[df_tmp.total.idxmax()]))) for
         df_tmp in dfs],
        columns=['max', 'last', 'trainset_size'])
    return {suffix: df.mean().to_dict()}


def toks_per_doc():
    config = ConfigParser()
    config.read(path.join(path.dirname(__file__), "../../path.conf"))
    jsonfile = path.join(config["DEFAULT"]["datapath"], config["DEFAULT"]["sentsplit_export_file"])
    with open(jsonfile) as f:
        tokens_in_doc = {doc_name_:
                             sum((len(sent) for sent in sents.values()))
                         for doc_name_, sents in json.load(f).items()}

    return tokens_in_doc


def col_indices(file, col, index_by_toknumber=True):
    doc_name = re.match(r'^.*?/(\w{,6})-', file)[1]
    if index_by_toknumber:
        tokens_in_doc = toks_per_doc()
        df = pd.read_csv(file).assign(
            train_tokens=lambda d: d.train_docs.apply(
                lambda x: sum((tokens_in_doc[doc] for doc in eval(x)))))
        series = pd.Series(dict(zip(df.train_tokens, df[col])))
    else:
        series = pd.read_csv(file)['col']
    return doc_name, series


def export_all_runs_as_fig(suffix, col='total'):
    df = pd.DataFrame(dict((col_indices(file, col) for file in
                            get_files_for_suffix(suffix)))).sort_index(axis=1).interpolate()
    ax = df.plot(colormap="tab20")
    ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    ax.grid(axis='y')
    ax.get_figure().savefig(f"out/__ALL__{suffix}.png", bbox_inches='tight')


if __name__ == '__main__':
    suffixes = [basename(dsrfile)[3:] for dsrfile in glob('out/DSR*.csv')]

    df = pd.DataFrame.from_dict(ChainMap(*[get_values(suffix) for suffix in suffixes]), orient='index')
    df = df.round(4).sort_values('trainset_size', ascending=False).sort_values('max')

    best_index = df.iloc[-1].name
    print(df.iloc[-1])

    with Pool() as p:
        p.map(export_all_runs_as_fig, df.index)
