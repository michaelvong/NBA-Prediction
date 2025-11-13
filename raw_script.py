import pandas as pd
import json
import time
import os
from datetime import datetime
from nba_api.stats.endpoints import LeagueGameFinder
from nba_api.stats.endpoints import BoxScoreAdvancedV3, BoxScoreFourFactorsV3, BoxScoreMiscV3, BoxScoreScoringV3, \
    BoxScoreTraditionalV3, BoxScoreUsageV3

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_progress(progress_id_file):
    """Load processed game IDs from progress file."""
    # Make path relative to script directory
    progress_path = os.path.join(SCRIPT_DIR, progress_id_file)
    try:
        with open(progress_path, 'r') as f:
            # Convert all IDs to strings for consistent comparison
            return set(str(game_id) for game_id in json.load(f))
    except FileNotFoundError:
        return set()


def save_progress(progress_id_file, processed_ids):
    """Save processed game IDs to progress file."""
    # Make path relative to script directory
    progress_path = os.path.join(SCRIPT_DIR, progress_id_file)
    with open(progress_path, 'w') as f:
        # Convert to list of strings
        json.dump(list(processed_ids), f)


def get_game_list(season, season_type):
    """Fetch all games for a given season."""
    print(f"Fetching game list for season {season}, type {season_type}...")
    gamefinder = LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type,
        league_id_nullable='00'
    )
    games = gamefinder.get_data_frames()[0]
    return games


def fetch_boxscore_data(game_id):
    """Fetch data from all box score endpoints for a single game."""
    time.sleep(3.5)  # Rate limiting

    data = {}

    # Define which columns to keep from each endpoint
    columns_to_keep = {
        'traditional': ['gameId', 'teamId', 'teamCity', 'teamName', 'teamTricode', 'teamSlug',
                        'fieldGoalsMade', 'fieldGoalsAttempted', 'fieldGoalsPercentage',
                        'threePointersMade', 'threePointersAttempted', 'threePointersPercentage',
                        'freeThrowsMade', 'freeThrowsAttempted', 'freeThrowsPercentage',
                        'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal',
                        'assists', 'steals', 'blocks', 'turnovers', 'foulsPersonal',
                        'points', 'plusMinusPoints'],
        'advanced': ['teamId',  # Only teamId for merging
                     'estimatedOffensiveRating', 'offensiveRating', 'estimatedDefensiveRating',
                     'defensiveRating', 'estimatedNetRating', 'netRating', 'assistPercentage',
                     'assistToTurnover', 'assistRatio', 'offensiveReboundPercentage',
                     'defensiveReboundPercentage', 'reboundPercentage',
                     'estimatedTeamTurnoverPercentage', 'turnoverRatio',
                     'effectiveFieldGoalPercentage', 'trueShootingPercentage',
                     'usagePercentage', 'estimatedUsagePercentage', 'estimatedPace',
                     'pace', 'pacePer40', 'possessions', 'PIE'],
        'fourfactors': ['teamId',  # Only teamId for merging
                        'effectiveFieldGoalPercentage', 'freeThrowAttemptRate',
                        'teamTurnoverPercentage', 'offensiveReboundPercentage',
                        'oppEffectiveFieldGoalPercentage', 'oppFreeThrowAttemptRate',
                        'oppTeamTurnoverPercentage', 'oppOffensiveReboundPercentage'],
        'misc': ['teamId',  # Only teamId for merging
                 'pointsOffTurnovers', 'pointsSecondChance', 'pointsFastBreak',
                 'pointsPaint', 'oppPointsOffTurnovers', 'oppPointsSecondChance',
                 'oppPointsFastBreak', 'oppPointsPaint', 'blocks', 'foulsPersonal',
                 'foulsDrawn'],
        'scoring': ['teamId',  # Only teamId for merging
                    'percentageFieldGoalsAttempted2pt', 'percentageFieldGoalsAttempted3pt',
                    'percentagePoints2pt', 'percentagePointsMidrange2pt',
                    'percentagePoints3pt', 'percentagePointsFastBreak',
                    'percentagePointsFreeThrow', 'percentagePointsOffTurnovers',
                    'percentagePointsPaint', 'percentageAssisted2pt',
                    'percentageUnassisted2pt', 'percentageAssisted3pt',
                    'percentageUnassisted3pt', 'percentageAssistedFGM',
                    'percentageUnassistedFGM'],
        'usage': ['teamId',  # Only teamId for merging
                  'percentagePersonalFoulsDrawn']
    }

    try:
        # BoxScoreTraditionalV3
        trad = BoxScoreTraditionalV3(game_id=game_id)
        trad_df = trad.team_stats.get_data_frame()
        # Keep only specified columns that exist
        cols = [c for c in columns_to_keep['traditional'] if c in trad_df.columns]
        data['traditional'] = trad_df[cols]

        # BoxScoreAdvancedV3
        adv = BoxScoreAdvancedV3(game_id=game_id)
        adv_df = adv.team_stats.get_data_frame()
        cols = [c for c in columns_to_keep['advanced'] if c in adv_df.columns]
        data['advanced'] = adv_df[cols]

        # BoxScoreFourFactorsV3
        four = BoxScoreFourFactorsV3(game_id=game_id)
        four_df = four.team_stats.get_data_frame()
        cols = [c for c in columns_to_keep['fourfactors'] if c in four_df.columns]
        data['fourfactors'] = four_df[cols]

        # BoxScoreMiscV3
        misc = BoxScoreMiscV3(game_id=game_id)
        misc_df = misc.team_stats.get_data_frame()
        cols = [c for c in columns_to_keep['misc'] if c in misc_df.columns]
        data['misc'] = misc_df[cols]

        # BoxScoreScoringV3
        scoring = BoxScoreScoringV3(game_id=game_id)
        scoring_df = scoring.team_stats.get_data_frame()
        cols = [c for c in columns_to_keep['scoring'] if c in scoring_df.columns]
        data['scoring'] = scoring_df[cols]

        # BoxScoreUsageV3
        usage = BoxScoreUsageV3(game_id=game_id)
        usage_df = usage.team_stats.get_data_frame()
        cols = [c for c in columns_to_keep['usage'] if c in usage_df.columns]
        data['usage'] = usage_df[cols]

        return data

    except Exception as e:
        print(f"Error fetching data for game {game_id}: {e}")
        return None


def merge_boxscore_data(data, game_id):
    """Merge all box score data for both teams in a game."""
    if not data:
        return None

    # Start with traditional stats (which has all identifier columns)
    merged = data['traditional'].copy()

    # Merge other endpoints - they only have teamId and their specific stats
    for key in ['advanced', 'fourfactors', 'misc', 'scoring', 'usage']:
        df = data[key].copy()

        # Merge on teamId
        # Add suffix to duplicate column names (e.g., if 'blocks' or 'usagePercentage' appear multiple times)
        merged = merged.merge(df, on='teamId', how='left', suffixes=('', f'_{key}'))

    return merged


def process_game(game_id, game_info):
    """Process a single game and return formatted data for both teams."""
    print(f"Processing game {game_id}...")

    boxscore_data = fetch_boxscore_data(game_id)
    if boxscore_data is None:
        return None

    merged_data = merge_boxscore_data(boxscore_data, game_id)
    if merged_data is None or len(merged_data) < 2:
        return None

    # Get game info
    home_team_id = game_info['home_team_id']
    away_team_id = game_info['away_team_id']
    game_date = game_info['game_date']
    season = game_info['season']
    home_team_score = game_info['home_score']
    away_team_score = game_info['away_score']

    # Convert merged_data to dictionary for easier access
    team_stats = {}
    for _, row in merged_data.iterrows():
        team_id = row['teamId']
        team_stats[team_id] = row.to_dict()

    result_rows = []

    # Create two rows: one from home perspective, one from away perspective
    for team_id in [home_team_id, away_team_id]:
        if team_id not in team_stats:
            continue

        team_data = team_stats[team_id]
        team_name = team_data['teamName']

        # Determine if home or away
        is_home = 1 if team_id == home_team_id else 0

        # Determine opponent
        opponent_id = away_team_id if is_home == 1 else home_team_id

        if opponent_id not in team_stats:
            continue

        opponent_data = team_stats[opponent_id]
        opponent_name = opponent_data['teamName']

        # Determine win/loss
        team_score = home_team_score if is_home == 1 else away_team_score
        opp_score = away_team_score if is_home == 1 else home_team_score
        win = 1 if team_score > opp_score else 0

        # Create result row with required features first
        result_row = {
            'game_id': game_id,
            'season': season,
            'game_date': game_date,
            'team_id': team_id,
            'team_name': team_name,
            'opponent_team_id': opponent_id,
            'opponent_team_name': opponent_name,
            'home': is_home,
            'win': win
        }

        # Add all team stats with 'team_' prefix
        for col in team_data.keys():
            if col not in ['gameId', 'teamId', 'teamName', 'teamCity', 'teamTricode']:
                result_row[f'team_{col}'] = team_data[col]

        # Add all opponent stats with 'opp_' prefix
        for col in opponent_data.keys():
            if col not in ['gameId', 'teamId', 'teamName', 'teamCity', 'teamTricode']:
                result_row[f'opp_{col}'] = opponent_data[col]

        result_rows.append(result_row)

    return pd.DataFrame(result_rows)


def main(season, season_type='Regular Season', progress_id_file='progress_ids.json',
         output_file='nba_team_stats.csv'):
    """
    Main function to collect NBA team statistics.

    Parameters:
    -----------
    season : str
        Season in format 'YYYY-YY' (e.g., '2023-24')
    season_type : str
        Type of season ('Regular Season', 'Playoffs', etc.)
    progress_id_file : str
        File to track processed game IDs
    output_file : str
        Final output CSV file
    """

    # Load progress
    processed_ids = load_progress(progress_id_file)

    # Try to load existing output file
    output_path = os.path.join(SCRIPT_DIR, output_file)
    try:
        all_data = pd.read_csv(output_path)
        print(f"Loaded {len(all_data)} previously processed rows from {output_path}")
    except FileNotFoundError:
        all_data = pd.DataFrame()

    print(f"Loaded {len(processed_ids)} previously processed game IDs")

    # Get game list
    games_df = get_game_list(season, season_type)

    # Create game info dictionary (one entry per game, not per team)
    game_dict = {}
    for _, row in games_df.iterrows():
        # Convert game_id to string for consistent comparison
        game_id = str(row['GAME_ID'])

        if game_id not in game_dict:
            # Determine home/away based on MATCHUP format
            matchup = row['MATCHUP']
            if ' @ ' in matchup:
                # This team is away
                is_home = False
            else:
                # This team is home
                is_home = True

            if is_home:
                game_dict[game_id] = {
                    'home_team_id': row['TEAM_ID'],
                    'away_team_id': None,
                    'home_score': row['PTS'],
                    'away_score': None,
                    'game_date': row['GAME_DATE'],
                    'season': season
                }
            else:
                if game_id in game_dict:
                    game_dict[game_id]['away_team_id'] = row['TEAM_ID']
                    game_dict[game_id]['away_score'] = row['PTS']
                else:
                    game_dict[game_id] = {
                        'home_team_id': None,
                        'away_team_id': row['TEAM_ID'],
                        'home_score': None,
                        'away_score': row['PTS'],
                        'game_date': row['GAME_DATE'],
                        'season': season
                    }
        else:
            # Fill in the missing team info
            matchup = row['MATCHUP']
            if ' @ ' in matchup:
                game_dict[game_id]['away_team_id'] = row['TEAM_ID']
                game_dict[game_id]['away_score'] = row['PTS']
            else:
                game_dict[game_id]['home_team_id'] = row['TEAM_ID']
                game_dict[game_id]['home_score'] = row['PTS']

    # Filter to unprocessed games - now both are strings
    unprocessed_games = {gid: info for gid, info in game_dict.items() if gid not in processed_ids}

    print(f"Total games: {len(game_dict)}")
    print(f"Unprocessed games: {len(unprocessed_games)}")

    # Debug: Show a few processed IDs
    if processed_ids:
        print(f"Sample processed IDs: {list(processed_ids)[:3]}")
    if game_dict:
        print(f"Sample game IDs from API: {list(game_dict.keys())[:3]}")

    # Process each game with keyboard interrupt handling
    try:
        for idx, (game_id, game_info) in enumerate(unprocessed_games.items(), 1):
            try:
                print(f"[{idx}/{len(unprocessed_games)}] Processing game {game_id}")

                game_data = process_game(game_id, game_info)

                if game_data is not None:
                    # Append to all_data
                    all_data = pd.concat([all_data, game_data], ignore_index=True)

                    # Mark as processed (as string)
                    processed_ids.add(str(game_id))

                    # Save progress after every game
                    save_progress(progress_id_file, processed_ids)
                    all_data.to_csv(output_path, index=False)
                    print(f"Progress saved: {len(processed_ids)} games processed")

            except KeyboardInterrupt:
                print("\n\nKeyboard interrupt detected! Saving progress immediately...")
                raise
            except Exception as e:
                print(f"Error processing game {game_id}: {e}")
                continue

    except KeyboardInterrupt:
        print("Saving all progress before exit...")
        save_progress(progress_id_file, processed_ids)
        all_data.to_csv(output_path, index=False)
        print(f"\n✓ Progress saved!")
        print(f"✓ Total rows: {len(all_data)}")
        print(f"✓ Total games processed: {len(processed_ids)}")
        print(f"✓ Data saved to: {output_path}")
        print(f"✓ Game IDs saved to: {os.path.join(SCRIPT_DIR, progress_id_file)}")
        print("\nYou can resume by running the script again with the same parameters.")
        return all_data

    # Final save (normal completion)
    save_progress(progress_id_file, processed_ids)
    all_data.to_csv(output_path, index=False)
    print(f"\nData collection complete!")
    print(f"Total rows: {len(all_data)}")
    print(f"Total games processed: {len(processed_ids)}")
    print(f"Output saved to: {output_path}")

    return all_data


if __name__ == "__main__":
    # Example usage
    df = main(
        season='2023-24',
        season_type='Regular Season',
        progress_id_file='progress_ids.json',
        output_file='nba_team_stats_2023_24.csv'
    )