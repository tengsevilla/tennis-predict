import pandas as pd
import datetime
import urllib.error
import os

MODEL_DIR = os.getenv("MODEL_DIR", "models/")

def fetch_and_process_data(years_back=5):
    base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{}.csv"
    current_year = datetime.datetime.now().year
    start_year = max(2016, current_year - years_back)

    os.makedirs(MODEL_DIR, exist_ok=True)

    dataframes = []
    fetched_from_github = []
    loaded_from_cache = []

    for year in range(start_year, current_year + 1):
        cached_path = os.path.join(MODEL_DIR, f"atp_{year}.csv")

        # Historical years never change — use cache if available.
        # Always re-fetch the current year since it is still being updated.
        if year < current_year and os.path.exists(cached_path):
            print(f"[{year}] Loading from cache...")
            df = pd.read_csv(cached_path)
            loaded_from_cache.append(year)
        else:
            url = base_url.format(year)
            print(f"[{year}] Fetching from GitHub...")
            try:
                df = pd.read_csv(url)
                df.to_csv(cached_path, index=False)
                fetched_from_github.append(year)
                print(f"[{year}] Fetched and cached {len(df)} rows.")
            except urllib.error.HTTPError as e:
                print(f"[{year}] HTTP {e.code} — ", end="")
                if os.path.exists(cached_path):
                    df = pd.read_csv(cached_path)
                    loaded_from_cache.append(year)
                    print("falling back to cache.")
                else:
                    print("no cache available, skipping.")
                    continue
            except Exception as e:
                print(f"[{year}] Error: {e} — skipping.")
                continue

        dataframes.append(df)

    if not dataframes:
        print("No data fetched.")
        return None, None

    combined_df = pd.concat(dataframes, ignore_index=True)
    initial_rows = len(combined_df)

    cleaned_df = combined_df.dropna(subset=['winner_rank', 'loser_rank'])
    final_rows = len(cleaned_df)

    # Freshness check
    tourney_dates = pd.to_datetime(cleaned_df['tourney_date'], format='%Y%m%d', errors='coerce')
    most_recent = tourney_dates.max()
    data_age_days = int((datetime.datetime.now() - most_recent).days) if pd.notna(most_recent) else None

    output_file = os.path.join(MODEL_DIR, "cleaned_atp_data.csv")
    print(f"Saving {final_rows} rows to {output_file}...")
    cleaned_df.to_csv(output_file, index=False)

    metadata = {
        'most_recent_match': most_recent.strftime('%Y-%m-%d') if pd.notna(most_recent) else 'unknown',
        'data_age_days': data_age_days,
        'total_rows': final_rows,
        'dropped_rows': initial_rows - final_rows,
        'years_range': f"{start_year}-{current_year}",
        'fetched_from_github': fetched_from_github,
        'loaded_from_cache': loaded_from_cache,
    }

    print(f"Done! Most recent match: {metadata['most_recent_match']} ({data_age_days} days ago).")
    return cleaned_df, metadata


if __name__ == "__main__":
    fetch_and_process_data()
