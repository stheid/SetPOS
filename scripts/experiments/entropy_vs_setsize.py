from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import exponential
from scipy.stats import entropy

from setpos.tagger import create_g_alpha_beta, create_ubop_predictor

if __name__ == '__main__':
    K, n = 10, 1000000

    A = exponential(size=(n, K))
    A = A / A.sum(axis=1)[:, None]
    entropys = pd.Series(np.apply_along_axis(entropy, 1, A))

    g = create_g_alpha_beta(1, 1, K)
    ubop = create_ubop_predictor(g)

    setsize = pd.Series(np.apply_along_axis(partial(ubop, labels=np.arange(K)), 1, A)).apply(len)

    df = pd.DataFrame([entropys, setsize], index=['entropy', 'setsize']).T
    sns.scatterplot(x='entropy', y='setsize', data=df)
    plt.show()
