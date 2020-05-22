import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import norm


def nudge_points(df, iterations=100, rate=.2, max_move=.1, seed=None):
    """
    x,y, size must be given
    :param df:
    :return: x,y will be modified and old x,y will be moved to x,y old
    """
    if seed is not None:
        np.random.seed(seed)

    # copy posiotions
    df = df.assign(x_old=df.x, y_old=df.y)

    has_moved = True
    i = 0

    while i < iterations and has_moved:
        i += 1
        has_moved = False

        for i1, _ in df.iterrows():
            for i2, p2 in df.loc[i1 + 1:].iterrows():
                # re-get point in case it has been updated
                p1 = df.loc[i1]
                p1_v = p1[['x', 'y']].to_numpy() + np.random.uniform(0, 1e-7, 2)
                p2_v = p2[['x', 'y']].to_numpy()

                # calculate overlap
                min_dist = ((p1['size'] + p2['size']) / 2)
                dist = p1_v - p2_v
                overlap = min_dist - norm(dist)

                if overlap <= 0:
                    # no overlap
                    continue

                # calculate intended movement
                ind_move = dist / norm(dist) * overlap * rate

                for sign, idx, p in zip((-1, 1), (i1, i2), (p1, p2)):
                    curr_move = p[['x_old', 'y_old']].to_numpy() - p[['x', 'y']].to_numpy()

                    # calculate allowed movement
                    total_move = curr_move + sign * ind_move
                    allowed_total_move = total_move / norm(total_move) * min(max_move, norm(total_move))

                    if norm(allowed_total_move - curr_move) == 0:
                        continue
                    has_moved = True
                    # print(i1, i2, idx, allowed_total_move)
                    df.loc[idx, ['x', 'y']] = p[['x_old', 'y_old']].to_numpy() + allowed_total_move
    return df


if __name__ == '__main__':
    np.random.seed(2)
    df = pd.DataFrame([(1, 1, 1), (1, 1, 1), (1, 1, 1), (3, 4, 1), (3, 1, 10)], columns=['x', 'y', 'val'])
    df = df.assign(size=.05 + (df.val - df.val.min()) / (df.val.max() - df.val.min()) * .0)

    df = nudge_points(df, iterations=100, rate=1, max_move=.05)

    sns.scatterplot(x='x', y='y', size='val', sizes=(100, 200), data=df)
    plt.show()
