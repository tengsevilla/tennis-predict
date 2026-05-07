import pandas as pd
import datetime
import urllib.request
import urllib.error

def fetch_and_process_data():
    base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{}.csv"
    current_year = datetime.datetime.now().year

    dataframes = []

    for year in range(2016, current_year + 1):
        url = base_url.format(year)
        print(f"Fetching data for year {year} from {url}...")
        try:
            # Read CSV directly from URL
            df = pd.read_csv(url)
            dataframes.append(df)
            print(f"Successfully loaded {len(df)} rows for {year}.")
        except urllib.error.HTTPError as e:
            print(f"Failed to fetch data for {year}: HTTP Error {e.code}")
        except Exception as e:
            print(f"Failed to fetch data for {year}: {e}")

    if not dataframes:
        print("No data fetched.")
        return

    print("Concatenating dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True)

    initial_rows = len(combined_df)
    print(f"Total rows before cleaning: {initial_rows}")

    # Drop rows where 'winner_rank' or 'loser_rank' is null
    cleaned_df = combined_df.dropna(subset=['winner_rank', 'loser_rank'])

    final_rows = len(cleaned_df)
    print(f"Total rows after cleaning: {final_rows} (Dropped {initial_rows - final_rows} rows)")

    # Save the cleaned output
    output_file = "cleaned_atp_data.csv"
    print(f"Saving to {output_file}...")
    cleaned_df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    fetch_and_process_data()
