import pandas as pd
import io
import os
from geopy.geocoders import Nominatim
from unidecode import unidecode
from tqdm import tqdm

geolocator = Nominatim(user_agent="geoapiExercises")

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'coordinates_with_country.csv')
df = pd.read_csv(csv_path)

def get_country(row):
    if row['country'] == 'Unknown':
        coord = f"{row['latitude']}, {row['longitude']}"
        try:
            # Perform reverse geocoding
            location = geolocator.reverse(coord, exactly_one=True)
            address = location.raw['address']
            country = address.get('country', 'Unknown')  # Get the country, or 'Unknown' if not found
            row['country'] = unidecode(country)
        except Exception as e:
            row['country'] = 'Unknown'
            print(f"\nError retrieving country for {coord}")
    return row

tqdm.pandas(desc="Retrieving countries")

df = df.progress_apply(get_country, axis=1)
# df['ID'] = range(len(df))
# df = df[['ID', 'latitude', 'longitude', 'country']]
df.to_csv('coordinates_with_country2.csv', index=False)

print("New CSV with countries and ID column has been created.")