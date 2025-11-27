import pandas as pd
import json
import os

# ------------------------------------------
# CONFIGS
# ------------------------------------------
CSV_FILE = "nba_team_stats_2024_25.csv"
OUTPUT_JSON = "progress_ids.json"

def load_game_ids(csv_file):
    """Load csv and extract game_id column."""
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find CSV file: {csv_file}")
        return []

    if "game_id" not in df.columns:
        print("❌ ERROR: 'game_id' column not found in CSV.")
        return []

    game_ids = df["game_id"].astype(str).dropna().unique().tolist()
    return game_ids

def format_game_ids(game_ids):
    """Add '00' prefix to each game id."""
    return [f"00{gid}" for gid in game_ids]

def save_raw_array(data, output_file):
    """Save ONLY an array into the JSON file (no wrapper object)."""
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Saved raw array with {len(data)} IDs to {output_file}")

def main():
    game_ids = load_game_ids(CSV_FILE)
    if not game_ids:
        return

    formatted = format_game_ids(game_ids)
    save_raw_array(formatted, OUTPUT_JSON)

if __name__ == "__main__":
    main()
