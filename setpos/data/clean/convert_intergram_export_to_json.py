import re
from collections import OrderedDict
from logging import warning

from setpos.data.clean.io_ import dump, load

tagset = set()


def is_punkt(tok):
    return tok in {":", "~", "…", '.;', "blankline", "/", ",", "•", '¶', '¶…', "ǂ", "|"} \
           or tok.startswith("$") \
           or tok.endswith(":") \
           or all([c == '.' for c in tok])


def is_sentbound(tok, tags):
    if list(tags.keys()) == ['XY']:
        return True
    if not tags:
        if re.match("[ivx]+:?\d*|\d+", tok) or tok == 'blankline':
            return True
    return False


def sentsplit_and_orthocorrect_doc(title, doc_text, enable_sent_splitting=False):
    global tagset
    doc, sentence = dict(), dict()

    s_i = 1
    t_i = 1
    for token, tag in re.findall("(\S+) \(\[(.*?)\]\)", doc_text):
        tags = OrderedDict(((t, float(score)) for t, score in re.findall('(\S+):(\d*.\d+)', tag)))

        # if its a word token (not interpunction)
        if not (enable_sent_splitting and is_sentbound(token, tags)):
            if (not tags or 'OA' in tags.keys()) and is_punkt(token) or any((t.startswith("$") for t in tags.keys())):
                # orthotags
                tags = {'$.': 1.}
            if not tags:
                tags = dict(OA=1.)
                # warn for untagged words
                warning(f"Untagged word '{token}' in document [{title}]")
            elif 'OA' in tags.keys():
                warning(f'"{list(tags.keys())}"-tagged word "{token}" in document [{title}]')
            else:
                # all other tags remain unchanged
                pass

            # add token to sencence
            sentence[f"t{t_i:0>4}"] = [token, tags]
            tagset = tagset.union(tags.keys())
            t_i += 1

        # finish sentence
        # if XY tag, or token contains at leas a number and is only made up by numbers (roman) and colon signs)
        # skip empty sentences
        if enable_sent_splitting and is_sentbound(token, tags) and sentence:
            doc[f"s{s_i:0>3}"] = sentence
            sentence = dict()
            s_i += 1
            t_i = 1
    else:
        doc[f"s{s_i:0>3}"] = sentence

    return doc


if __name__ == '__main__':
    docs = load()
    docs = {f"{title}": sentsplit_and_orthocorrect_doc(title, text) for title, text in docs.items()}
    print(len(tagset), sorted(tagset))
    dump(docs)
