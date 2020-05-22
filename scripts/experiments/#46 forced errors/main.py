from collections import Counter
from itertools import chain

import pandas as pd

if __name__ == '__main__':
    df = pd.read_excel('forced_errors.xlsx')
    df.sort_values(['target'], inplace=True)
    df = df.groupby(['target'], as_index=False).constrainedtags.agg(
        {
            "confusions": lambda x: sorted(Counter(chain.from_iterable(map(eval, x))).items(), key=lambda i: i[1],
                                           reverse=True),
            'count': 'size'
        })

    df.to_excel('forced_errors_agg.xlsx')
