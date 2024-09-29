import csv
import geoip2.database
from tqdm import tqdm

# Load the GeoLite2 database
geoip_reader = geoip2.database.Reader('GeoLite2-Country.mmdb')

# Function to get the country from latitude and longitude
def get_country(latitude, longitude):
    try:
        response = geoip_reader.country(latitude, longitude)
        return response.country.name if response.country else 'Unknown'
    except Exception as e:
        print(f"Error retrieving country for {latitude}, {longitude}: {e}")
        return 'Unknown'

# File paths
input_file = 'coordinates.csv'
output_file = 'image_country.csv'

# Open the input CSV file and read it
with open(input_file, mode='r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    
    # Skip the header row in the input file
    headers = next(reader)

    # Get the total number of rows for the progress bar
    total_rows = sum(1 for _ in reader)  # Count total rows excluding header
    infile.seek(0)  # Reset the file pointer to the start of the file
    next(reader)  # Skip the header again

    # Create a progress bar for processing
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)

        # Write the headers for the output CSV
        writer.writerow(['Image ID', 'Country'])

        # Process each row and get the country
        for idx, row in tqdm(enumerate(reader), total=total_rows, desc='Processing'):
            try:
                latitude, longitude = float(row[0]), float(row[1])  # Convert to float for geoip2
                country = get_country(latitude, longitude)

                # Write the Photo ID (image name) and country to the output CSV
                writer.writerow([idx, country])
            except ValueError as e:
                print(f"Value error for row {idx}: {row} - {e}")

print("CSV with Photo ID and Country has been created.")
