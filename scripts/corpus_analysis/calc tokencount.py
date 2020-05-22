import pandas as pd

from setpos.data.split import load, join_to_sents

if __name__ == '__main__':
    X, y, groups = load()  # [l[:3000] for l in load()]  #

    results = []
    sents = join_to_sents(X, y)

    for sent in sents:
        new_sents_count_full_carth = pd.Series([len(tags) * 1. for _, tags in sent]).prod()
        new_sents_count_ambig_carth = pd.Series(
            [1.] + [max(1., len(list(filter(lambda v: v >= 1, list(tags.values())))))
                    for _, tags in sent]).prod()
        new_sents_count_each_tag_used = pd.Series([len(tags) for _, tags in sent]).max()

        results.append(dict(before=len(sent),
                            carth=new_sents_count_full_carth * len(sent),
                            ambig_carth=new_sents_count_ambig_carth * len(sent),
                            each_tag_used=new_sents_count_each_tag_used * len(sent)))

    df = pd.DataFrame(results)
    with pd.option_context('display.float_format', '{:,.0f}'.format):
        print(df.sum())
