import pandas as pd
from collections import Counter

from setpos.data.split import load
from matplotlib import pyplot as plt

if __name__ == '__main__':
    X, y, g = load()
    count = Counter(X[:, 1])
    series = pd.Series(count).sort_values()
    print(series)
    series = series[series < 100]
    series.hist(bins=50)
    plt.show()
