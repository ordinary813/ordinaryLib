import pandas as pd
import os

df = pd.read_csv('coordinates_with_country.csv')
counts = df['country'].value_counts()
counts.to_csv('country_counts.csv', sep=',')