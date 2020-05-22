import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from joblib import delayed, Parallel
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, StandardScaler

from scripts.corpus_metadata.probability_gains import toks_per_doc
from setpos.data.xmlparse import load
from setpos.util import nudge_points


def extract_cora_header(doc):
    soup = BeautifulSoup(doc, features="lxml")

    return soup.select_one("cora-header").attrs


if __name__ == '__main__':
    DEBUG = True
    docs = load()

    # create pandas dataframe for the documents
    df = pd.DataFrame(Parallel(n_jobs=-1)(delayed(extract_cora_header)(doc[:1000]) for doc in docs))
    df = df.astype(dict(century_id="int32", location_id="int32", text_type_id="int32"))

    cent = pd.read_csv('century.csv', index_col=0).to_dict('index')
    loc = pd.read_csv('location.csv', index_col=0).to_dict('index')

    df = df.assign(year=df.century_id.apply(lambda i: cent[i]['year']))
    df = df.assign(**{col: df.location_id.apply(lambda i: loc[i][col]) for col in ['lon', 'lat']})
    toks = toks_per_doc()
    df = df.assign(Tokencount=df.sigle.apply(lambda i: toks[i]))

    if DEBUG:
        df = df.assign(Datasource=df.sigle.apply(lambda x: 'REN' if x.startswith('REN') else 'other'))
        # plot geo normalized locations as a sanity check
        plt.imshow(mpimg.imread('map.png'), aspect=1.64, interpolation='spline16', alpha=.2,
                   extent=[5.625, 16.875, 48.9, 55.93])

        df = df.assign(
            size=.03 + (df.Tokencount - df.Tokencount.min()) / (df.Tokencount.max() - df.Tokencount.min()) * .4,
            x=df.lon, y=df.lat)
        df = nudge_points(df, iterations=40, rate=.4, max_move=.12, seed=1)
        df = df.assign(Longitude=df.x, Latitude=df.y, Year=df.year)

        ax = sns.scatterplot(x='Longitude', y='Latitude', hue='Year', size='Tokencount', sizes=(15, 200),
                             style='Datasource', data=df)
        plt.xlim(df.lon.min() - 1, 16.875)
        plt.ylim(df.lat.min() - .3, df.lat.max() + .3)
        # for i, r in df.iterrows():
        #    ax.text(r.lon, r.lat, r.sigle)
        fig = ax.get_figure()
        # fig.set_size_inches((9.6, 7))
        fig.savefig("dataset_distribution.pdf", bbox_inches='tight')
        plt.clf()

    df = df[['sigle', 'name', 'text_type_id', 'year', 'lon', 'lat', 'Tokencount']]
    print('total tokens:', df.Tokencount.sum())

    # normalize data
    data = df[['year', 'lon', 'lat']]
    # first we make their mean corelate
    norm_geo = scale(data[['lon', 'lat']], with_std=False)
    # than we calculate their combined scaling and
    geo_scaler = StandardScaler(with_mean=False)
    geo_scaler.fit(norm_geo.reshape(-1, 1))
    norm_geo = geo_scaler.transform(norm_geo)
    norm_year = scale(data.year.to_frame())
    data = np.hstack([norm_year, norm_geo])

    # add normalized data to df
    df = df.assign(year_norm=norm_year, lon_norm=norm_geo[:, 0], lat_norm=norm_geo[:, 1])
    df.to_csv('corpus.metadata.csv')

    # plot dimensionality reduced distances
    pca = PCA(n_components=2)
    X = pca.fit_transform(data)
    print("Explained variance", pca.explained_variance_ratio_)
    ax = sns.scatterplot(data=pd.DataFrame(X), x=0, y=1)
    for i, v in df.sigle.items():
        ax.text(X[i, 0], X[i, 1], v)
    # plt.show()
