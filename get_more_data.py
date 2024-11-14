import kagglehub
import pandas as pd
import json
from datetime import datetime

# Load the dataset
path = kagglehub.dataset_download("aparoski/runescape-grand-exchange-data")
df = pd.read_csv(path + "\Runescape_GE_Analytics_2024-09-18\Runescape_Item_Prices.csv")

# Convert date to datetime object
df['date'] = pd.to_datetime(df['date'])

# Filter for the last 5 years (from today's date)
today = datetime.today()
five_years_ago = today.replace(year=today.year - 5)
df = df[df['date'] >= five_years_ago]

# List of item IDs to filter
item_ids = ["554", "555", "556", "557", "558", "559", "560", "561", "562", "563", "564", "565", "566"]

# Filter for the relevant item IDs
df = df[df['id'].astype(str).isin(item_ids)]

# Remove rows with NaN values in 'price' or 'volume'
df = df.dropna(subset=['price', 'volume'])

# Initialize the result dictionary
result = {}

# Get the union of all dates across all items
all_dates = df['date'].unique()
all_dates.sort()

# Process each item ID
for item_id in item_ids:
    item_data = df[df['id'] == int(item_id)]
    
    # Group data by date (in case there are multiple entries per day)
    grouped = item_data.groupby('date').agg(
        lowprice=('price', 'min'),
        highprice=('price', 'max'),
        lowprice_volume=('volume', 'min'),
        highprice_volume=('volume', 'max')
    ).reset_index()

    # Reindex the grouped data to include all possible dates and align timesteps
    grouped.set_index('date', inplace=True)
    grouped = grouped.reindex(all_dates, method=None)
    
    # Convert the data into the desired format, filling NaN values if missing
    result[item_id] = grouped[['lowprice', 'lowprice_volume', 'highprice', 'highprice_volume']].fillna(method='ffill').values.tolist()

# Store the result in a JSON file
output_path = 'runescape_data.json'
with open(output_path, 'w') as f:
    json.dump(result, f, indent=4)

print(f"Data has been saved to {output_path}")
