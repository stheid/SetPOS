import json
from collections import OrderedDict

from setpos.data.split import MCInDocSplitter, load, dump
from setpos.tagger import CoreNLPTagger
from scripts.evaluation.general import SEED


def merge_target_pred_dicts(t, p):
    d = OrderedDict([('::' + k + '::', v) for k, v in json.loads(t, object_pairs_hook=OrderedDict).items()])
    d.update(**json.loads(p, object_pairs_hook=OrderedDict))
    return json.dumps(d)


if __name__ == '__main__':
    toks, tags, groups = load(tag_prefix_masks=[])  # [l[:3000] for l in load(tag_prefix_masks=[])]

    # train - test split
    train, test = next(MCInDocSplitter(seed=SEED).split(toks, tags, groups))

    # take the training data for train/eval cross-validation
    Xtrain, ytrain, _ = [l[train] for l in [toks, tags, groups]]

    # take the training data for train/eval cross-validation
    toks_e, tags_e, groups_e = [l[test] for l in [toks, tags, groups]]

    train_mask = groups_e != 'REN4'

    clf = CoreNLPTagger(**dict(
        corenlp_train_params=
        ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1",
         "-rareWordMinFeatureThresh", "1", "-rareWordThresh", "6",
         "-sigmaSquared",
         "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch",
         "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wordTag(0,-1),words(-3,2)",
         '-lang', 'english']))
    clf.fit(Xtrain, ytrain)
    pred = clf.setpredict(toks_e[~train_mask])

    target_and_pred = [merge_target_pred_dicts(t, p) for t, p in zip(tags_e[~train_mask], pred)]

    with open('ren4.html', 'w') as f:
        dump([(toks_e[~train_mask], target_and_pred)], [f], as_dataset=True,
             tagsed_sents_tostr_kws=dict(tags_to_string='html'))
