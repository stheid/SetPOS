import json
from collections import OrderedDict
from configparser import ConfigParser
from itertools import repeat, groupby
from os import path
from typing import Sequence, Tuple, Union

import numpy as np

config = ConfigParser()
project_root = path.join(path.dirname(__file__), "../../../")
config.read(path.join(project_root, "path.conf"))
ifile = path.join(project_root, config["DEFAULT"]["datapath"], config["DEFAULT"]["sentsplit_export_file"])

html_head = '''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.2.1/js/bootstrap.min.js"></script>
</head>
<body><div class="container mt-5" >
<h3>Bremer Urkunden 1351-1400</h3>
    <p class="text-justify">'''

html_foot = '''</p></div>
<script>
$(function () {
  $('[data-toggle="tooltip"]').tooltip()
})
</script>
</body>
</html>'''


def is_masked(tags, prefixes):
    return all(not tag.startswith(prefix) for tag in json.loads(tags).keys() for prefix in prefixes)


def filtertags(toks, tags, tag_prefix_masks):
    return zip(*[(tok, tags) for tok, tags in zip(toks, tags) if is_masked(tags, tag_prefix_masks)])


def load(file=ifile, tag_prefix_masks=(), n: int = None):
    with open(file) as f:
        data = json.load(f)

    toks_tags_groups = []
    i = 0
    for title, doc in data.items():
        doc = [list(sent.values()) for sent in doc.values()]
        toks, tags = sents_to_dataset(doc, i)
        toks, tags = filtertags(toks, tags, tag_prefix_masks)
        i += len(toks)
        toks_tags_groups.extend(list(zip(toks, tags, repeat(title, len(toks)))))
    # return toks, tags, groups
    if n:
        return [np.array(l)[:n] for l in zip(*toks_tags_groups)]
    return [np.array(l) for l in zip(*toks_tags_groups)]


def tagged_sents_tostring(sents, dlm="/", word_dlm=' ', sent_dlm='\n', tags_to_string='first'):
    lines = []
    for sent in sents:
        words = []
        for tok, tags in sent:
            if tags_to_string == 'html':
                tags_as_table = f'<table>{"".join([f"<tr><td>{k}</td><td>{v:.2f}</td></tr>" for k, v in tags.items()])}</table>'
                word = f'<span data-toggle="tooltip" data-html="true" title="{tags_as_table}">{tok}</span>'
                words.append(word)
            elif tags_to_string == 'dict':
                words.append(tok + '{' + ', '.join((f'{k}:{v:.2f}' for k, v in tags.items())) + '}')
            elif tags_to_string == None:
                words.append(tok)
            else:
                words.append(tok + dlm + list(tags.keys())[0])
        lines.append(word_dlm.join(words))

    result = ''
    if tags_to_string == 'html':
        result += html_head
        glue = '</p>\n<p class="text-justify">'
        result += glue.join(lines)
        result += html_foot
    else:
        result += sent_dlm.join(lines)
    return result


def sents_to_dataset(sents: Union[Sequence[Sequence[Tuple[str, dict]]], Sequence[Sequence[str]]], offset: int = 0):
    """
    list of sentences, where each sent is a list of tok+tags pairs.
    Merges sentences and puts a sentencenumber on each toktag pair
    :param offset:
    :param sents:
    :return:
    """
    try:
        toks, tags = zip(*[((i, tok), json.dumps(tags))
                           for i, sent in enumerate(sents, offset)
                           for tok, tags in sent])
        return toks, tags
    except ValueError:
        return [((i, tok))
                for i, sent in enumerate(sents, offset)
                for tok in sent]


def join_to_sents(X: Sequence[str], y=None, augment=False):
    """
    takes list of tokens and list of taglists and merges them to sentences
    the merging is done by groupby, which means it is merging only consecutive sentence ids, this means that ill-labled sentences like

    1111 2222 1111

    are parsed correctly as three sentences.
    :param X: sequence of (sentid,tok) tuples
    :param y:
    :return:
    """
    sents = []
    y = [json.loads(y_, object_pairs_hook=OrderedDict) for y_ in y] if y is not None else y
    it = zip(X, y) if y is not None else iter(X)
    sentid = lambda e: e[0][0] if y is not None else e[0]

    def get_tag(tags, i=0):
        if y is not None and isinstance(y[0], dict):
            tags = list(tags.items())
            return dict([tags[np.clip([i], -len(tags), len(tags) - 1)[0]]])
        else:
            return tags

    tok_tag = lambda e: (e[0][1], e[1]) if y is not None else e[1]

    # group by sentid
    for _, sent in groupby(it, key=sentid):
        sent = list(sent)
        if augment and y is not None:
            max_setsize = np.max([len(tags) for _, tags in sent])
            for i in range(max_setsize):
                sents.append([(tok[1], get_tag(tags, -i)) for tok, tags in sent])
        else:
            sents.append([tok_tag(elem) for elem in sent])
    return sents


def dump(data: Sequence[Union[Tuple[Sequence], Tuple[Sequence, Sequence]]], filehandles=None,
         as_dataset=False, augment=False, tagsed_sents_tostr_kws=None):
    if tagsed_sents_tostr_kws is None:
        tagsed_sents_tostr_kws = dict()
    if filehandles is None:
        if len(data) != 3:
            raise ValueError("provide filehandles or provide 3 datasets (whole,train,test)")
        filehandles = [path.join(config["DEFAULT"]["datapath"],
                                 config["DEFAULT"]["train_eval_split_file"]).format(s)
                       for s in ["dataset", "train", "eval"]]

    for sents, file in zip(data, filehandles):
        if as_dataset:
            if len(sents) == 1:
                # unpack
                toks = sents[0]
                # add dummy tags if they are not provided
                sents = toks, repeat('{"OA":1.0}', len(toks))
            sents = join_to_sents(*sents, augment=augment)
        if isinstance(file, str):
            with open(file, "w") as f:
                print(tagged_sents_tostring(sents, **tagsed_sents_tostr_kws), file=f)
        else:
            print(tagged_sents_tostring(sents, **tagsed_sents_tostr_kws), file=file)


if __name__ == '__main__':
    ds = sents_to_dataset([[("ich", ["NN"]), ("mag", ["NN"]), ("züge", ["NN"])],
                           [("i", ["NN"]), ("like", ["NN"]), ("trains", ["NN"])]])
    print(ds)
    print(join_to_sents(*ds))

    X, y, g = load()
    print("\n".join((" ".join(sent) for sent in join_to_sents(X[g == 'SOS1']))))
