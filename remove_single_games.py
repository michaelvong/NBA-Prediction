import pandas as pd
import json
import os

# ------------------------------------------
# CONFIGS
# ------------------------------------------
CSV_FILE = "nba_team_stats_2022_23.csv"
OUTPUT_CSV = "nba_team_stats_2022_23_clean.csv"
PROGRESS_JSON = "progress_ids.json"

def load_progress_ids(json_file):
    """Load existing progress IDs array from JSON."""
    if not os.path.exists(json_file):
        return []

    try:
        with open(json_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                print(f"⚠️ JSON content is not an array, ignoring: {json_file}")
                return []
    except json.JSONDecodeError:
        print(f"⚠️ JSON decode error, ignoring: {json_file}")
        return []

def save_progress_ids(ids, json_file):
    """Save progress IDs array to JSON (raw array, no curly braces)."""
    with open(json_file, "w") as f:
        json.dump(ids, f, indent=4)

def main():
    # Load CSV
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"❌ ERROR: CSV file not found: {CSV_FILE}")
        return

    if "game_id" not in df.columns:
        print("❌ ERROR: 'game_id' column not found in CSV.")
        return

    # Count rows per game_id
    counts = df["game_id"].value_counts()

    # Find game_ids that do not have exactly 2 rows
    incomplete_game_ids = counts[counts != 2].index.tolist()

    if incomplete_game_ids:
        print("⚠️ Game IDs with incomplete rows (not exactly 2):")
        for gid in incomplete_game_ids:
            print(gid)
    else:
        print("✅ All game_ids have exactly 2 rows!")

    # Remove rows with incomplete game_ids
    df_clean = df[~df["game_id"].isin(incomplete_game_ids)]

    # Save cleaned CSV
    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Cleaned CSV saved to {OUTPUT_CSV} (removed {len(incomplete_game_ids)} game_ids)")

    # --------------------------
    # Update progress_ids.json
    # --------------------------
    progress_ids = load_progress_ids(PROGRESS_JSON)
    # Remove any IDs that match incomplete game_ids (with or without "00" prefix)
    cleaned_progress_ids = [
        pid for pid in progress_ids
        if pid[2:] not in incomplete_game_ids  # remove "00" prefix
    ]
    save_progress_ids(cleaned_progress_ids, PROGRESS_JSON)
    print(f"✅ Updated {PROGRESS_JSON} (removed {len(progress_ids) - len(cleaned_progress_ids)} IDs)")

if __name__ == "__main__":
    main()
