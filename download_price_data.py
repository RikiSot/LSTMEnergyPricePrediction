from pathlib import Path

import pandas as pd
import requests
from datetime import datetime, timedelta


def convert_response(data):
    """
    Converts the API response data to a pandas DataFrame.

    Parameters
    ----------
    data : dict
       The API response data.

    Returns
    -------
    df : DataFrame
       A DataFrame containing the data from the API response.
    """
    # Extract the 'values' list from the response data
    values_list = data['included'][0]['attributes']['values']

    # Create a DataFrame from the 'values' list
    df = pd.DataFrame(values_list)

    # Convert the 'datetime' column to datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Set the 'timeStamp' column as the index
    df.set_index('datetime', inplace=True)
    df.index.rename('timeStamp', inplace=True)

    # Rename the 'value' column to 'value'
    df = df[['value']]

    return df


def get_price_data(start_date: datetime, end_date: datetime):
    # TODO: if there are different timezones handle index apart
    # Define the base URL
    base_url = "https://apidatos.ree.es/en/datos/mercados/precios-mercados-tiempo-real"

    # Convert dates to ISO 8601 format
    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()

    # Construct the full URL
    full_url = f"{base_url}?start_date={start_date_str}&end_date={end_date_str}&time_trunc=hour"

    # Send the GET request
    response = requests.get(full_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        df = convert_response(data)
        return df
    else:
        print(f'Request failed with status code {response.status_code}, {response.text}')


def download_training_data(year: int, path: str | Path):
    """
    Downloads training data for a given year and writes it to a CSV file.

    Args:
       year: The year for which to download data.
       path: The path to the CSV file where the data will be written.

    Returns:
       None
    """
    # Define the start and end dates for the given year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23, 59, 59)

    # Define the batch size
    batch_size = timedelta(days=25)

    # Initialize an empty DataFrame to hold the data
    data = pd.DataFrame()

    # Iterate over the batches
    current_date = start_date
    while current_date <= end_date:
        # Define the start and end dates for the current batch
        batch_start = current_date
        batch_end = min(current_date + batch_size, end_date)

        # Download the price data for the current batch
        df = get_price_data(batch_start, batch_end)

        # Append the data to the DataFrame
        data = pd.concat([data, df])

        # Move to the next batch
        current_date += batch_size

    # Write the data to a CSV file
    data.to_csv(path)


if __name__ == '__main__':
    data_path = Path('./data/price_data.csv')
    training_year = datetime.now().year - 1
    download_training_data(year=training_year, path=data_path)
