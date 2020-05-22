from multiprocessing.pool import Pool

import pandas as pd
import requests

historical_mapping = dict(ravensberg="Bielefeld")


def get_geo_loc(city):
    city = city.lower()
    if city in historical_mapping:
        city = historical_mapping[city]

    query = f'https://nominatim.openstreetmap.org/search?q={city}&format=json'
    try:
        first_matched_city = requests.get(query).json()[0]
    except IndexError:
        print('"{city}" not found in osm')
        return [0, 0]
    return [first_matched_city[i] for i in ['lon', 'lat']]


if __name__ == '__main__':
    df = pd.read_csv('location_old.csv', names=["city"], sep=";", header=None, index_col=0)

    with Pool(32) as p:
        coords = p.map(get_geo_loc, df.city)
    # we use the same indices as in df, as we will join on index.
    # Otherwise it will be zero indexed and the dataframes indices start with 1.
    coords = pd.DataFrame(coords, columns=['lon', 'lat'], index=df.index)

    df = df.join(coords)
    df.to_csv('location.csv')
