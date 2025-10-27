"""
NBA Game-by-Game Rolling Statistics Builder
Fetches game-by-game data with rolling team and player statistics.
Each game has TWO rows - one for each team with opponent stats.
Includes comprehensive progress saving to resume after interruptions.
NOW WITH VARIANCE FEATURES FOR L3, L5, L10 WINDOWS
"""
import random
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder, boxscoreadvancedv2, boxscoretraditionalv2

def convert_min_to_numeric(min_str):
    """
    Convert 'MM:SS' string to float minutes.
    """
    if isinstance(min_str, str) and ':' in min_str:
        mins, secs = min_str.split(':')
        return float(mins) + float(secs)/60
    try:
        return float(min_str)
    except:
        return 0.0


def fetch_games_for_season(season='2023-24', season_type='Regular Season'):
    """
    Fetch all games for a season.
    Parameters:
    -----------
    season : str
        Season in format 'YYYY-YY'
    season_type : str
        'Regular Season' or 'Playoffs'
    Returns:
    --------
    pd.DataFrame
        DataFrame with all games
    """
    print(f"\nFetching games for {season} {season_type}...")

    game_finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type,
        league_id_nullable='00'  # NBA only
    )

    games = game_finder.get_data_frames()[0]

    # Sort by game date
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games = games.sort_values('GAME_DATE').reset_index(drop=True)

    print(f"  ✓ Found {len(games)} team-game records ({len(games)//2} unique games)")

    return games


def fetch_box_score_traditional(game_id):
    """Fetch traditional box score for a game."""
    try:
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        player_stats = box.get_data_frames()[0]
        team_stats = box.get_data_frames()[1]
        return player_stats, team_stats
    except Exception as e:
        print(f"    Error fetching traditional box score for {game_id}: {str(e)}")
        return None, None


def fetch_box_score_advanced(game_id):
    """Fetch advanced box score for a game."""
    try:
        box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
        player_stats = box.get_data_frames()[0]
        team_stats = box.get_data_frames()[1]
        return player_stats, team_stats
    except Exception as e:
        print(f"    Error fetching advanced box score for {game_id}: {str(e)}")
        return None, None


def safe_variance(series, min_count=1):
    """
    Calculate variance with minimum period of 1.
    Returns 0 if only 1 value, otherwise calculates sample variance.
    """
    if len(series) < min_count:
        return 0.0
    elif len(series) == 1:
        return 0.0
    else:
        return series.var(ddof=1)


def calculate_rolling_stats(game_data, team_id, game_index):
    """
    Calculate rolling statistics up to (but not including) the current game.
    Includes both simple averages and NBA's possession-weighted methodology.
    Also includes last 3, 5, and 10 games averages and VARIANCE for all stats.
    Parameters:
    -----------
    game_data : pd.DataFrame
        All game data for the season
    team_id : int
        Team ID
    game_index : int
        Current game index in the season
    Returns:
    --------
    dict
        Dictionary of rolling statistics
    """
    # Get all previous games for this team
    prev_games = game_data[
        (game_data['TEAM_ID'] == team_id) &
        (game_data['GAME_INDEX'] < game_index)
    ]

    if len(prev_games) == 0:
        # First game of season - return 0/default values (NO NaN)
        base_stats = {
            # Simple averages (existing)
            'rolling_ortg': 0,
            'rolling_drtg': 0,
            'rolling_netrtg': 0,
            'rolling_pace': 0,
            'rolling_efg': 0,
            'rolling_ts': 0,
            'rolling_ast_pct': 0,
            'rolling_ast_to': 0,
            'rolling_ast_ratio': 0,
            'rolling_oreb_pct': 0,
            'rolling_dreb_pct': 0,
            'rolling_reb_pct': 0,
            'rolling_tov_pct': 0,
            'rolling_fg_pct': 0,
            'rolling_fg3_pct': 0,
            'rolling_ft_pct': 0,
            'rolling_pts': 0,
            'rolling_starter_per': 0,
            'rolling_bench_per': 0,
            'rolling_starter_usg': 0,
            'rolling_bench_usg': 0,
            'wins_last_10': 0,

            # NBA methodology (possession-weighted)
            'nba_ortg': 0,
            'nba_drtg': 0,
            'nba_netrtg': 0,
            'nba_pace': 0,
            'nba_efg': 0,
            'nba_ts': 0,
            'nba_sefg': 0,
            'nba_ast_pct': 0,
            'nba_tov_pct': 0,
            'nba_oreb_pct': 0,
            'nba_dreb_pct': 0,
            'nba_reb_pct': 0,
        }

        # Add L3, L5, L10 means and variances (all set to 0 for first game)
        variance_stats = ['pts', 'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta',
                         'oreb', 'dreb', 'reb', 'ast', 'tov',
                         'fg_pct', 'fg3_pct', 'ft_pct',
                         'ortg', 'drtg', 'netrtg', 'pace',
                         'efg', 'ts', 'ast_pct', 'ast_to', 'ast_ratio',
                         'oreb_pct', 'dreb_pct', 'reb_pct', 'tov_pct',
                         'opp_netrtg', 'opp_ortg', 'opp_drtg',
                         'opp_efg', 'opp_tov_pct', 'opp_oreb_pct',
                         'opp_dreb_pct', 'opp_fta', 'opp_fga',
                         'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta',
                         'oreb', 'dreb', 'opp_oreb', 'opp_dreb', 'poss',
                         'three_pt_dep_eff', 'two_pt_rate_eff',
                         'ft_to_fg_ratio', 'pct_pts_from_threes',
                         'oreb_battle_won', 'team_reb_share',
                         'pts_per_poss_fg']

        for window in ['l3', 'l5', 'l10']:
            for stat in variance_stats:
                base_stats[f'{window}_{stat}'] = 0
                base_stats[f'{window}_{stat}_var'] = 0

        return base_stats

    # Calculate simple rolling averages (existing)
    stats = {
        'rolling_ortg': prev_games['OFF_RATING'].mean() if 'OFF_RATING' in prev_games else 0,
        'rolling_drtg': prev_games['DEF_RATING'].mean() if 'DEF_RATING' in prev_games else 0,
        'rolling_netrtg': prev_games['NET_RATING'].mean() if 'NET_RATING' in prev_games else 0,
        'rolling_pace': prev_games['PACE'].mean() if 'PACE' in prev_games else 0,
        'rolling_efg': prev_games['EFG_PCT'].mean() if 'EFG_PCT' in prev_games else 0,
        'rolling_ts': prev_games['TS_PCT'].mean() if 'TS_PCT' in prev_games else 0,
        'rolling_ast_pct': prev_games['AST_PCT'].mean() if 'AST_PCT' in prev_games else 0,
        'rolling_ast_to': prev_games['AST_TO'].mean() if 'AST_TO' in prev_games else 0,
        'rolling_ast_ratio': prev_games['AST_RATIO'].mean() if 'AST_RATIO' in prev_games else 0,
        'rolling_oreb_pct': prev_games['OREB_PCT'].mean() if 'OREB_PCT' in prev_games else 0,
        'rolling_dreb_pct': prev_games['DREB_PCT'].mean() if 'DREB_PCT' in prev_games else 0,
        'rolling_reb_pct': prev_games['REB_PCT'].mean() if 'REB_PCT' in prev_games else 0,
        'rolling_tov_pct': prev_games['TOV_PCT'].mean() if 'TOV_PCT' in prev_games else 0,
        'rolling_fg_pct': prev_games['FG_PCT'].mean() if 'FG_PCT' in prev_games else 0,
        'rolling_fg3_pct': prev_games['FG3_PCT'].mean() if 'FG3_PCT' in prev_games else 0,
        'rolling_ft_pct': prev_games['FT_PCT'].mean() if 'FT_PCT' in prev_games else 0,
        'rolling_pts': prev_games['PTS'].mean() if 'PTS' in prev_games else 0,
        'rolling_starter_per': prev_games['STARTER_PER_AVG'].mean() if 'STARTER_PER_AVG' in prev_games else 0,
        'rolling_bench_per': prev_games['BENCH_PER_AVG'].mean() if 'BENCH_PER_AVG' in prev_games else 0,
        'rolling_starter_usg': prev_games['STARTER_USG_AVG'].mean() if 'STARTER_USG_AVG' in prev_games else 0,
        'rolling_bench_usg': prev_games['BENCH_USG_AVG'].mean() if 'BENCH_USG_AVG' in prev_games else 0,
    }

    # Wins in last 10 games
    last_10 = prev_games.tail(10)
    stats['wins_last_10'] = last_10['WL'].sum()

    # Define column mapping for variance calculations
    col_mapping = {
        'pts': 'PTS', 'fgm': 'FGM', 'fga': 'FGA', 'fg3m': 'FG3M', 'fg3a': 'FG3A',
        'ftm': 'FTM', 'fta': 'FTA', 'oreb': 'OREB', 'dreb': 'DREB', 'reb': 'REB',
        'ast': 'AST', 'tov': 'TOV', 'fg_pct': 'FG_PCT', 'fg3_pct': 'FG3_PCT',
        'ft_pct': 'FT_PCT', 'ortg': 'OFF_RATING', 'drtg': 'DEF_RATING',
        'netrtg': 'NET_RATING', 'pace': 'PACE', 'efg': 'EFG_PCT', 'ts': 'TS_PCT',
        'ast_pct': 'AST_PCT', 'ast_to': 'AST_TO', 'ast_ratio': 'AST_RATIO',
        'oreb_pct': 'OREB_PCT', 'dreb_pct': 'DREB_PCT', 'reb_pct': 'REB_PCT',
        'tov_pct': 'TOV_PCT', 'poss': 'POSS',
        'opp_netrtg': 'OPP_NET_RATING', 'opp_ortg': 'OPP_OFF_RATING',
        'opp_drtg': 'OPP_DEF_RATING', 'opp_efg': 'OPP_EFG_PCT',
        'opp_tov_pct': 'OPP_TOV_PCT', 'opp_oreb_pct': 'OPP_OREB_PCT',
        'opp_dreb_pct': 'OPP_DREB_PCT', 'opp_fta': 'OPP_FTA', 'opp_fga': 'OPP_FGA',
        'opp_oreb': 'OPP_OREB', 'opp_dreb': 'OPP_DREB'
    }

    # Calculate L3, L5, L10 means and variances
    for window_size, window_name in [(3, 'l3'), (5, 'l5'), (10, 'l10')]:
        window_games = prev_games.tail(window_size)

        if len(window_games) > 0:
            # Basic stats means and variances
            for stat_name, col_name in col_mapping.items():
                if col_name in window_games.columns:
                    stats[f'{window_name}_{stat_name}'] = window_games[col_name].mean()
                    stats[f'{window_name}_{stat_name}_var'] = safe_variance(window_games[col_name])
                else:
                    stats[f'{window_name}_{stat_name}'] = 0
                    stats[f'{window_name}_{stat_name}_var'] = 0

            # Derived features for this window
            avg_fgm = stats[f'{window_name}_fgm']
            avg_fga = stats[f'{window_name}_fga']
            avg_fg3m = stats[f'{window_name}_fg3m']
            avg_fg3a = stats[f'{window_name}_fg3a']
            avg_ftm = stats[f'{window_name}_ftm']
            avg_oreb = stats[f'{window_name}_oreb']
            avg_dreb = stats[f'{window_name}_dreb']
            avg_opp_oreb = stats[f'{window_name}_opp_oreb']
            avg_opp_dreb = stats[f'{window_name}_opp_dreb']
            avg_pts = stats[f'{window_name}_pts']
            avg_poss = stats[f'{window_name}_poss']
            avg_fg_pct = stats[f'{window_name}_fg_pct']
            avg_fg3_pct = stats[f'{window_name}_fg3_pct']

            # Calculate derived features
            if avg_fga > 0:
                stats[f'{window_name}_three_pt_dep_eff'] = (avg_fg3a / avg_fga) * avg_fg3_pct
                stats[f'{window_name}_two_pt_rate_eff'] = ((avg_fga - avg_fg3a) / avg_fga) * avg_fg_pct
            else:
                stats[f'{window_name}_three_pt_dep_eff'] = 0
                stats[f'{window_name}_two_pt_rate_eff'] = 0

            if (avg_fgm + avg_fg3m) > 0:
                stats[f'{window_name}_ft_to_fg_ratio'] = avg_ftm / (avg_fgm + avg_fg3m)
            else:
                stats[f'{window_name}_ft_to_fg_ratio'] = 0

            if avg_pts > 0:
                stats[f'{window_name}_pct_pts_from_threes'] = (avg_fg3m * 3) / avg_pts
            else:
                stats[f'{window_name}_pct_pts_from_threes'] = 0

            if avg_opp_dreb > 0:
                stats[f'{window_name}_oreb_battle_won'] = avg_oreb / avg_opp_dreb
            else:
                stats[f'{window_name}_oreb_battle_won'] = 0

            total_rebs = avg_oreb + avg_dreb + avg_opp_oreb + avg_opp_dreb
            if total_rebs > 0:
                stats[f'{window_name}_team_reb_share'] = (avg_oreb + avg_dreb) / total_rebs
            else:
                stats[f'{window_name}_team_reb_share'] = 0

            if avg_poss > 0:
                stats[f'{window_name}_pts_per_poss_fg'] = (avg_fgm + 0.5 * avg_fg3m) / avg_poss
            else:
                stats[f'{window_name}_pts_per_poss_fg'] = 0

            # Variances for derived features
            if len(window_games) > 1:
                three_pt_dep_series = (window_games['FG3A'] / window_games['FGA']) * window_games['FG3_PCT']
                stats[f'{window_name}_three_pt_dep_eff_var'] = safe_variance(three_pt_dep_series)

                two_pt_rate_series = ((window_games['FGA'] - window_games['FG3A']) / window_games['FGA']) * window_games['FG_PCT']
                stats[f'{window_name}_two_pt_rate_eff_var'] = safe_variance(two_pt_rate_series)

                ft_to_fg_series = window_games['FTM'] / (window_games['FGM'] + window_games['FG3M'])
                stats[f'{window_name}_ft_to_fg_ratio_var'] = safe_variance(ft_to_fg_series)

                pct_pts_threes_series = (window_games['FG3M'] * 3) / window_games['PTS']
                stats[f'{window_name}_pct_pts_from_threes_var'] = safe_variance(pct_pts_threes_series)

                oreb_battle_series = window_games['OREB'] / window_games['OPP_DREB']
                stats[f'{window_name}_oreb_battle_won_var'] = safe_variance(oreb_battle_series)

                total_rebs_series = window_games['OREB'] + window_games['DREB'] + window_games['OPP_OREB'] + window_games['OPP_DREB']
                team_reb_share_series = (window_games['OREB'] + window_games['DREB']) / total_rebs_series
                stats[f'{window_name}_team_reb_share_var'] = safe_variance(team_reb_share_series)

                pts_per_poss_series = (window_games['FGM'] + 0.5 * window_games['FG3M']) / window_games['POSS']
                stats[f'{window_name}_pts_per_poss_fg_var'] = safe_variance(pts_per_poss_series)
            else:
                stats[f'{window_name}_three_pt_dep_eff_var'] = 0
                stats[f'{window_name}_two_pt_rate_eff_var'] = 0
                stats[f'{window_name}_ft_to_fg_ratio_var'] = 0
                stats[f'{window_name}_pct_pts_from_threes_var'] = 0
                stats[f'{window_name}_oreb_battle_won_var'] = 0
                stats[f'{window_name}_team_reb_share_var'] = 0
                stats[f'{window_name}_pts_per_poss_fg_var'] = 0
        else:
            # If no games in window, set everything to 0
            for stat_name in col_mapping.keys():
                stats[f'{window_name}_{stat_name}'] = 0
                stats[f'{window_name}_{stat_name}_var'] = 0

            # Set derived features to 0
            derived_features = ['three_pt_dep_eff', 'two_pt_rate_eff', 'ft_to_fg_ratio',
                              'pct_pts_from_threes', 'oreb_battle_won', 'team_reb_share',
                              'pts_per_poss_fg']
            for feat in derived_features:
                stats[f'{window_name}_{feat}'] = 0
                stats[f'{window_name}_{feat}_var'] = 0

    # NBA Methodology - Possession-weighted cumulative stats
    # Get cumulative totals
    total_poss = prev_games['POSS'].sum() if 'POSS' in prev_games.columns else 0
    total_pts = prev_games['PTS'].sum() if 'PTS' in prev_games.columns else 0
    total_opp_pts = prev_games['OPP_PTS'].sum() if 'OPP_PTS' in prev_games.columns else 0

    # Offensive Rating = Points per 100 possessions
    stats['nba_ortg'] = (total_pts / total_poss * 100) if total_poss > 0 else 0

    # Defensive Rating = Opponent points per 100 possessions
    stats['nba_drtg'] = (total_opp_pts / total_poss * 100) if total_poss > 0 else 0

    # Net Rating = ORTG - DRTG
    stats['nba_netrtg'] = stats['nba_ortg'] - stats['nba_drtg']

    # Pace = Possessions per 48 minutes (weighted by minutes played)
    prev_games = prev_games.copy()
    prev_games['MIN_PLAYED_NUM'] = prev_games['MIN_PLAYED'].apply(convert_min_to_numeric)
    total_min = prev_games['MIN_PLAYED_NUM'].sum()
    stats['nba_pace'] = (total_poss / total_min * 48) if total_min > 0 else 0

    # EFG% = (FGM + 0.5 * 3PM) / FGA
    total_fgm = prev_games['FGM'].sum() if 'FGM' in prev_games.columns else 0
    total_fg3m = prev_games['FG3M'].sum() if 'FG3M' in prev_games.columns else 0
    total_fga = prev_games['FGA'].sum() if 'FGA' in prev_games.columns else 0
    stats['nba_efg'] = ((total_fgm + 0.5 * total_fg3m) / total_fga) if total_fga > 0 else 0

    # TS% = PTS / (2 * (FGA + 0.44 * FTA))
    total_fta = prev_games['FTA'].sum() if 'FTA' in prev_games.columns else 0
    ts_denominator = 2 * (total_fga + 0.44 * total_fta)
    stats['nba_ts'] = (total_pts / ts_denominator) if ts_denominator > 0 else 0

    # sEFG = Stabilized EFG (weighted average of TS% and EFG%)
    stats['nba_sefg'] = (0.6 * stats['nba_ts'] + 0.4 * stats['nba_efg']) if (stats['nba_ts'] > 0 or stats['nba_efg'] > 0) else 0

    # AST% = AST / FGM (team assists per made field goal)
    total_ast = prev_games['AST'].sum() if 'AST' in prev_games.columns else 0
    stats['nba_ast_pct'] = (total_ast / total_fgm * 100) if total_fgm > 0 else 0

    # TOV% = TOV / (FGA + 0.44 * FTA + TOV)
    total_tov = prev_games['TOV'].sum() if 'TOV' in prev_games.columns else 0
    tov_denominator = total_fga + 0.44 * total_fta + total_tov
    stats['nba_tov_pct'] = (total_tov / tov_denominator * 100) if tov_denominator > 0 else 0

    # OREB% = OREB / (OREB + OPP_DREB)
    total_oreb = prev_games['OREB'].sum() if 'OREB' in prev_games.columns else 0
    total_opp_dreb = prev_games['OPP_DREB'].sum() if 'OPP_DREB' in prev_games.columns else 0
    oreb_denominator = total_oreb + total_opp_dreb
    stats['nba_oreb_pct'] = (total_oreb / oreb_denominator * 100) if oreb_denominator > 0 else 0

    # DREB% = DREB / (DREB + OPP_OREB)
    total_dreb = prev_games['DREB'].sum() if 'DREB' in prev_games.columns else 0
    total_opp_oreb = prev_games['OPP_OREB'].sum() if 'OPP_OREB' in prev_games.columns else 0
    dreb_denominator = total_dreb + total_opp_oreb
    stats['nba_dreb_pct'] = (total_dreb / dreb_denominator * 100) if dreb_denominator > 0 else 0

    # REB% = (OREB + DREB) / (OREB + DREB + OPP_OREB + OPP_DREB)
    total_reb = total_oreb + total_dreb
    total_opp_reb = total_opp_oreb + total_opp_dreb
    reb_denominator = total_reb + total_opp_reb
    stats['nba_reb_pct'] = (total_reb / reb_denominator * 100) if reb_denominator > 0 else 0

    return stats


def process_game_boxscores(season='2023-24', season_type='Regular Season', max_games=None,
                           progress_id_file='nba_progress.csv',
                           progress_data_file='nba_game_data_progress_2023_24.csv'):
    """
    Process all games in a season to create game-by-game rolling statistics.
    Saves complete game data after each game to allow resuming.

    Parameters:
    -----------
    season : str
        Season in format 'YYYY-YY'
    season_type : str
        'Regular Season' or 'Playoffs'
    max_games : int, optional
        Limit number of games to process (for testing)
    progress_id_file : str
        File containing list of processed game IDs (one per line)
    progress_data_file : str
        File to save/load complete game data

    Returns:
    --------
    pd.DataFrame
        Game-by-game data with rolling statistics
    """
    # Get all games
    games = fetch_games_for_season(season, season_type)

    # CRITICAL: Convert GAME_ID to string in the games DataFrame itself
    games['GAME_ID'] = games['GAME_ID'].astype(str)

    # Get unique game IDs (now guaranteed to be strings)
    unique_games = games['GAME_ID'].unique()

    if max_games:
        unique_games = unique_games[:max_games]
        print(f"\n⚠ Processing only first {max_games} games for testing")

    # Load processed game IDs from nba_progress.csv (separate from game data)
    processed_game_ids = set()
    if os.path.exists(progress_id_file):
        print(f"\n✓ Found progress ID file: {progress_id_file}")
        try:
            # Read as plain text with dtype=str to preserve leading zeros
            progress_df = pd.read_csv(progress_id_file, dtype=str)

            # Check if file is empty or has no data
            if progress_df.empty or len(progress_df) == 0:
                print(f"⚠ Progress ID file is empty. Starting from scratch...")
            else:
                # Get the game_id column (assume first column or column named 'game_id' or 'GAME_ID')
                if 'game_id' in progress_df.columns:
                    processed_game_ids = set(progress_df['game_id'].dropna().astype(str).str.strip().str.zfill(10))
                elif 'GAME_ID' in progress_df.columns:
                    processed_game_ids = set(progress_df['GAME_ID'].dropna().astype(str).str.strip().str.zfill(10))
                else:
                    # Assume first column contains game IDs
                    processed_game_ids = set(progress_df.iloc[:, 0].dropna().astype(str).str.strip().str.zfill(10))

                print(f"✓ Loaded {len(processed_game_ids)} processed game IDs")
                print(f"✓ Sample processed game IDs: {sorted(list(processed_game_ids))[:5]}")
        except (pd.errors.EmptyDataError, IndexError) as e:
            print(f"⚠ Progress ID file is empty or malformed. Starting from scratch...")
            processed_game_ids = set()
    else:
        print(f"\n⚠ No progress ID file found: {progress_id_file}")

    # Load existing game data from nba_game_data_progress_2023_24.csv
    all_game_data = []
    if os.path.exists(progress_data_file):
        print(f"\n✓ Found game data file: {progress_data_file}")
        try:
            # Read with GAME_ID as string to preserve leading zeros
            existing_data = pd.read_csv(progress_data_file, dtype={'GAME_ID': str})

            # Check if file is empty
            if existing_data.empty or len(existing_data) == 0:
                print(f"⚠ Game data file is empty. Starting from scratch...")
                all_game_data = []
            else:
                # Ensure GAME_ID has leading zeros
                existing_data['GAME_ID'] = existing_data['GAME_ID'].str.zfill(10)

                print(f"✓ Loaded {len(existing_data)} game records")

                # Convert existing data to list of dicts
                all_game_data = existing_data.to_dict('records')
        except (pd.errors.EmptyDataError, ValueError) as e:
            print(f"⚠ Game data file is empty or malformed. Starting from scratch...")
            all_game_data = []
    else:
        print(f"\n⚠ No game data file found: {progress_data_file}")

    # Debug: Show sample of games to process
    print(f"\n✓ Sample game IDs to process: {unique_games[:5].tolist()}")
    for gid in unique_games[:3]:
        print(f"  - ID: '{gid}' | Length: {len(gid)}")

    print(f"\nProcessing {len(unique_games)} total games...")
    print(f"Already processed: {len(processed_game_ids)} games")
    print(f"Remaining: {len(unique_games) - len(processed_game_ids)} games")
    print("This will take a while due to API rate limits...\n")

    for idx, game_id in enumerate(unique_games):
        # Debug: Check first game
        if idx == 0:
            print(f"\n✓ First game_id: '{game_id}' | Type: {type(game_id)} | Length: {len(game_id)}")
            if processed_game_ids:
                sample_processed = list(processed_game_ids)[0]
                print(f"✓ Sample processed ID: '{sample_processed}' | Type: {type(sample_processed)} | Length: {len(sample_processed)}")
                print(f"✓ Are they equal? {game_id == sample_processed}")

        # Skip if already processed (both are strings with leading zeros)
        if game_id in processed_game_ids:
            if (idx + 1) % 50 == 0 or idx < 5:
                print(f"[{idx+1}/{len(unique_games)}] ⭐ Skipping already processed game: {game_id}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing game {idx+1}/{len(unique_games)}: {game_id}")
        print(f"{'='*60}")

        # Get game info from games dataframe (game_id is already string)
        game_info = games[games['GAME_ID'] == game_id]

        if len(game_info) != 2:
            print(f"  ⚠ Skipping game {game_id} - incomplete data")
            continue

        # Fetch box scores
        player_trad, team_trad = fetch_box_score_traditional(game_id)
        time.sleep(random.uniform(0.6, 0.8))  # Rate limiting

        player_adv, team_adv = fetch_box_score_advanced(game_id)
        time.sleep(random.uniform(0.6, 0.8))  # Rate limiting

        if player_trad is None or player_adv is None or team_trad is None or team_adv is None:
            print(f"  ⚠ Skipping game {game_id} - box score unavailable")
            continue

        # Process each team in the game
        game_records = []
        for _, team_game in game_info.iterrows():
            team_id = team_game['TEAM_ID']
            opp_id = game_info[game_info['TEAM_ID'] != team_id]['TEAM_ID'].values[0]

            # Get team box score stats - safely access columns
            team_trad_stats = team_trad[team_trad['TEAM_ID'] == team_id].iloc[0]
            team_adv_stats = team_adv[team_adv['TEAM_ID'] == team_id].iloc[0]

            # Get opponent stats
            opp_trad_stats = team_trad[team_trad['TEAM_ID'] == opp_id].iloc[0]
            opp_adv_stats = team_adv[team_adv['TEAM_ID'] == opp_id].iloc[0]

            # Helper function to safely get stat
            def safe_get(stats_df, col, default=None):
                return stats_df[col] if col in stats_df.index else default

            # Calculate AST_TO if not available (AST / TOV)
            ast_to = safe_get(team_adv_stats, 'AST_TO')
            if ast_to is None:
                ast = safe_get(team_trad_stats, 'AST', 0)
                tov = safe_get(team_trad_stats, 'TO', 1)
                ast_to = ast / tov if tov > 0 else None

            opp_ast_to = safe_get(opp_adv_stats, 'AST_TO')
            if opp_ast_to is None:
                opp_ast = safe_get(opp_trad_stats, 'AST', 0)
                opp_tov = safe_get(opp_trad_stats, 'TO', 1)
                opp_ast_to = opp_ast / opp_tov if opp_tov > 0 else None

            # Calculate current game derived features
            team_pts = safe_get(team_trad_stats, 'PTS', 0)
            team_fgm = safe_get(team_trad_stats, 'FGM', 0)
            team_fga = safe_get(team_trad_stats, 'FGA', 1)
            team_fg3m = safe_get(team_trad_stats, 'FG3M', 0)
            team_fg3a = safe_get(team_trad_stats, 'FG3A', 1)
            team_ftm = safe_get(team_trad_stats, 'FTM', 0)
            team_fta = safe_get(team_trad_stats, 'FTA', 1)
            team_oreb = safe_get(team_trad_stats, 'OREB', 0)
            team_dreb = safe_get(team_trad_stats, 'DREB', 0)
            team_fg_pct = safe_get(team_trad_stats, 'FG_PCT', 0)
            team_fg3_pct = safe_get(team_trad_stats, 'FG3_PCT', 0)
            team_poss = safe_get(team_adv_stats, 'POSS', 1)

            opp_pts = safe_get(opp_trad_stats, 'PTS', 0)
            opp_fgm = safe_get(opp_trad_stats, 'FGM', 0)
            opp_fga = safe_get(opp_trad_stats, 'FGA', 1)
            opp_fg3m = safe_get(opp_trad_stats, 'FG3M', 0)
            opp_ftm = safe_get(opp_trad_stats, 'FTM', 0)
            opp_fta = safe_get(opp_trad_stats, 'FTA', 1)
            opp_oreb = safe_get(opp_trad_stats, 'OREB', 0)
            opp_dreb = safe_get(opp_trad_stats, 'DREB', 0)

            # Derived features for current game
            three_pt_dependence_eff = (team_fg3a / team_fga) * team_fg3_pct if team_fga > 0 else None
            two_pt_attempt_rate_eff = ((team_fga - team_fg3a) / team_fga) * team_fg_pct if team_fga > 0 else None
            ft_to_fg_ratio = team_ftm / (team_fgm + team_fg3m) if (team_fgm + team_fg3m) > 0 else None
            pct_pts_from_threes = (team_fg3m * 3) / team_pts if team_pts > 0 else None
            oreb_battle_won = team_oreb / opp_dreb if opp_dreb > 0 else None
            team_reb_share = (team_oreb + team_dreb) / (team_oreb + team_dreb + opp_oreb + opp_dreb) if (team_oreb + team_dreb + opp_oreb + opp_dreb) > 0 else None
            fta_rate = team_fta / team_fga if team_fga > 0 else None
            opp_fta_rate = opp_fta / opp_fga if opp_fga > 0 else None
            pts_per_poss_fg = (team_fgm + 0.5 * team_fg3m) / team_poss if team_poss > 0 else None

            # Get player stats for this team
            team_players_trad = player_trad[player_trad['TEAM_ID'] == team_id].copy()
            team_players_adv = player_adv[player_adv['TEAM_ID'] == team_id].copy()

            # Merge player stats - select only available columns
            adv_cols_to_merge = ['PLAYER_ID']
            desired_adv_cols = ['E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING',
                                'AST_PCT', 'AST_TO', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT',
                                'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'USG_PCT',
                                'E_USG_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE']

            for col in desired_adv_cols:
                if col in team_players_adv.columns:
                    adv_cols_to_merge.append(col)

            team_players = team_players_trad.merge(
                team_players_adv[adv_cols_to_merge],
                on='PLAYER_ID',
                how='left'
            )

            # Identify starters (top 5 by minutes)
            team_players = team_players.sort_values('MIN', ascending=False)
            team_players['MIN_NUMERIC'] = team_players['MIN'].apply(
                lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 if isinstance(x, str) and ':' in x else 0
            )

            starters = team_players.head(5)
            bench = team_players.iloc[5:].head(3)  # Top 3 bench players by minutes

            # Calculate starter and bench averages
            starter_per_avg = starters['PIE'].mean() if len(starters) > 0 and 'PIE' in starters.columns else None
            bench_per_avg = bench['PIE'].mean() if len(bench) > 0 and 'PIE' in bench.columns else None
            starter_usg_avg = starters['USG_PCT'].mean() if len(starters) > 0 and 'USG_PCT' in starters.columns else None
            bench_usg_avg = bench['USG_PCT'].mean() if len(bench) > 0 and 'USG_PCT' in bench.columns else None

            # Create game record with safe column access
            game_record = {
                'GAME_ID': game_id,  # Already string with leading zeros
                'SEASON': season,
                'GAME_DATE': team_game['GAME_DATE'],
                'MATCHUP': team_game['MATCHUP'],
                'TEAM_ID': team_id,
                'TEAM_NAME': team_game['TEAM_NAME'],
                'OPP_TEAM_ID': opp_id,
                'IS_HOME': 1 if 'vs.' in team_game['MATCHUP'] else 0,
                'WL': 1 if team_game['WL'] == 'W' else 0,

                # Current game stats (not rolling) - need these for NBA methodology calculations
                'PTS': team_game['PTS'],
                'FG_PCT': team_game['FG_PCT'],
                'FG3_PCT': team_game['FG3_PCT'],
                'FT_PCT': team_game['FT_PCT'],
                'FGM': safe_get(team_trad_stats, 'FGM'),
                'FGA': safe_get(team_trad_stats, 'FGA'),
                'FG3M': safe_get(team_trad_stats, 'FG3M'),
                'FG3A': safe_get(team_trad_stats, 'FG3A'),
                'FTM': safe_get(team_trad_stats, 'FTM'),
                'FTA': safe_get(team_trad_stats, 'FTA'),
                'OREB': safe_get(team_trad_stats, 'OREB'),
                'DREB': safe_get(team_trad_stats, 'DREB'),
                'REB': safe_get(team_trad_stats, 'REB'),
                'AST': safe_get(team_trad_stats, 'AST'),
                'TOV': safe_get(team_trad_stats, 'TO'),
                'POSS': safe_get(team_adv_stats, 'POSS'),
                'MIN_PLAYED': safe_get(team_trad_stats, 'MIN'),
                'OFF_RATING': safe_get(team_adv_stats, 'OFF_RATING'),
                'DEF_RATING': safe_get(team_adv_stats, 'DEF_RATING'),
                'NET_RATING': safe_get(team_adv_stats, 'NET_RATING'),
                'PACE': safe_get(team_adv_stats, 'PACE'),
                'AST_PCT': safe_get(team_adv_stats, 'AST_PCT'),
                'AST_TO': ast_to,
                'AST_RATIO': safe_get(team_adv_stats, 'AST_RATIO'),
                'OREB_PCT': safe_get(team_adv_stats, 'OREB_PCT'),
                'DREB_PCT': safe_get(team_adv_stats, 'DREB_PCT'),
                'REB_PCT': safe_get(team_adv_stats, 'REB_PCT'),
                'TOV_PCT': safe_get(team_adv_stats, 'TM_TOV_PCT'),
                'EFG_PCT': safe_get(team_adv_stats, 'EFG_PCT'),
                'TS_PCT': safe_get(team_adv_stats, 'TS_PCT'),
                'STARTER_PER_AVG': starter_per_avg,
                'BENCH_PER_AVG': bench_per_avg,
                'STARTER_USG_AVG': starter_usg_avg,
                'BENCH_USG_AVG': bench_usg_avg,

                # Opponent current game stats (for NBA methodology calculations)
                'OPP_PTS': safe_get(opp_trad_stats, 'PTS'),
                'OPP_FGM': safe_get(opp_trad_stats, 'FGM'),
                'OPP_FGA': safe_get(opp_trad_stats, 'FGA'),
                'OPP_FG3M': safe_get(opp_trad_stats, 'FG3M'),
                'OPP_FTA': safe_get(opp_trad_stats, 'FTA'),
                'OPP_OREB': safe_get(opp_trad_stats, 'OREB'),
                'OPP_DREB': safe_get(opp_trad_stats, 'DREB'),
                'OPP_AST': safe_get(opp_trad_stats, 'AST'),
                'OPP_TOV': safe_get(opp_trad_stats, 'TO'),
                'OPP_OFF_RATING': safe_get(opp_adv_stats, 'OFF_RATING'),
                'OPP_DEF_RATING': safe_get(opp_adv_stats, 'DEF_RATING'),
                'OPP_NET_RATING': safe_get(opp_adv_stats, 'NET_RATING'),
                'OPP_PACE': safe_get(opp_adv_stats, 'PACE'),
                'OPP_AST_PCT': safe_get(opp_adv_stats, 'AST_PCT'),
                'OPP_AST_TO': opp_ast_to,
                'OPP_AST_RATIO': safe_get(opp_adv_stats, 'AST_RATIO'),
                'OPP_OREB_PCT': safe_get(opp_adv_stats, 'OREB_PCT'),
                'OPP_DREB_PCT': safe_get(opp_adv_stats, 'DREB_PCT'),
                'OPP_REB_PCT': safe_get(opp_adv_stats, 'REB_PCT'),
                'OPP_TOV_PCT': safe_get(opp_adv_stats, 'TM_TOV_PCT'),
                'OPP_EFG_PCT': safe_get(opp_adv_stats, 'EFG_PCT'),
                'OPP_TS_PCT': safe_get(opp_adv_stats, 'TS_PCT'),
                'OPP_FG_PCT': safe_get(opp_trad_stats, 'FG_PCT'),
                'OPP_FG3_PCT': safe_get(opp_trad_stats, 'FG3_PCT'),
                'OPP_FT_PCT': safe_get(opp_trad_stats, 'FT_PCT'),

                # Derived features (current game)
                'three_pt_dependence_eff': three_pt_dependence_eff,
                'two_pt_attempt_rate_eff': two_pt_attempt_rate_eff,
                'ft_to_fg_ratio': ft_to_fg_ratio,
                'pct_pts_from_threes': pct_pts_from_threes,
                'oreb_battle_won': oreb_battle_won,
                'team_reb_share': team_reb_share,
                'fta_rate': fta_rate,
                'opp_fta_rate': opp_fta_rate,
                'pts_per_poss_fg': pts_per_poss_fg,
            }

            game_records.append(game_record)
            all_game_data.append(game_record)

        # Add to processed set
        processed_game_ids.add(game_id)

        # Save progress to BOTH files after processing both teams for this game
        # 1. Save game data to nba_game_data_progress_2023_24.csv
        df_progress = pd.DataFrame(all_game_data)
        df_progress.to_csv(progress_data_file, index=False)

        # 2. Save processed game IDs to nba_progress.csv
        pd.DataFrame(sorted(list(processed_game_ids)), columns=['game_id']).to_csv(progress_id_file, index=False)

        print(f"  ✓ Progress saved: {len(all_game_data)} game records, {len(processed_game_ids)} unique games processed")

    # Create DataFrame from all collected data
    df = pd.DataFrame(all_game_data)

    # Add game index for rolling calculations
    df = df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
    df['GAME_INDEX'] = df.groupby('TEAM_ID').cumcount().astype(int)

    # Calculate rolling statistics
    print("\nCalculating rolling statistics...")
    rolling_stats_list = []

    for idx, row in df.iterrows():
        rolling_stats = calculate_rolling_stats(df, row['TEAM_ID'], row['GAME_INDEX'])
        rolling_stats_list.append(rolling_stats)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} rows")

    # Add rolling stats to dataframe
    rolling_df = pd.DataFrame(rolling_stats_list)
    df = pd.concat([df, rolling_df], axis=1)

    # Calculate tempo (pace * net_rating)
    df['rolling_tempo'] = df['rolling_pace'] * df['rolling_netrtg']

    # Calculate rolling point differential
    df['rolling_pts_diff'] = df['rolling_pts'] - df.groupby(['GAME_ID'])['rolling_pts'].transform(lambda x: x.iloc[::-1].values)

    # Calculate derived features for rolling stats
    print("\nCalculating derived rolling features...")

    # Calculate rolling derived features using cumulative values
    for idx, row in df.iterrows():
        # Get previous games for this team
        prev_games_df = df[(df['TEAM_ID'] == row['TEAM_ID']) & (df['GAME_INDEX'] < row['GAME_INDEX'])]

        if len(prev_games_df) > 0:
            # Rolling averages for derived features
            avg_fgm = prev_games_df['FGM'].mean()
            avg_fga = prev_games_df['FGA'].mean()
            avg_fg3m = prev_games_df['FG3M'].mean()
            avg_fg3a = prev_games_df['FG3A'].mean()
            avg_ftm = prev_games_df['FTM'].mean()
            avg_oreb = prev_games_df['OREB'].mean()
            avg_dreb = prev_games_df['DREB'].mean()
            avg_opp_oreb = prev_games_df['OPP_OREB'].mean()
            avg_opp_dreb = prev_games_df['OPP_DREB'].mean()
            avg_pts = prev_games_df['PTS'].mean()
            avg_poss = prev_games_df['POSS'].mean()
            avg_fg_pct = prev_games_df['FG_PCT'].mean()
            avg_fg3_pct = prev_games_df['FG3_PCT'].mean()
            avg_opp_fta = prev_games_df['OPP_FTA'].mean()
            avg_opp_fga = prev_games_df['OPP_FGA'].mean()

            # Calculate derived rolling features
            df.at[idx, 'rolling_three_pt_dep_eff'] = (avg_fg3a / avg_fga) * avg_fg3_pct if avg_fga > 0 else 0
            df.at[idx, 'rolling_two_pt_rate_eff'] = ((avg_fga - avg_fg3a) / avg_fga) * avg_fg_pct if avg_fga > 0 else 0
            df.at[idx, 'rolling_ft_to_fg_ratio'] = avg_ftm / (avg_fgm + avg_fg3m) if (avg_fgm + avg_fg3m) > 0 else 0
            df.at[idx, 'rolling_pct_pts_from_threes'] = (avg_fg3m * 3) / avg_pts if avg_pts > 0 else 0
            df.at[idx, 'rolling_oreb_battle_won'] = avg_oreb / avg_opp_dreb if avg_opp_dreb > 0 else 0
            df.at[idx, 'rolling_team_reb_share'] = (avg_oreb + avg_dreb) / (avg_oreb + avg_dreb + avg_opp_oreb + avg_opp_dreb) if (avg_oreb + avg_dreb + avg_opp_oreb + avg_opp_dreb) > 0 else 0
            df.at[idx, 'rolling_pts_per_poss_fg'] = (avg_fgm + 0.5 * avg_fg3m) / avg_poss if avg_poss > 0 else 0
            df.at[idx, 'rolling_opp_fta_rate'] = avg_opp_fta / avg_opp_fga if avg_opp_fga > 0 else 0
        else:
            df.at[idx, 'rolling_three_pt_dep_eff'] = 0
            df.at[idx, 'rolling_two_pt_rate_eff'] = 0
            df.at[idx, 'rolling_ft_to_fg_ratio'] = 0
            df.at[idx, 'rolling_pct_pts_from_threes'] = 0
            df.at[idx, 'rolling_oreb_battle_won'] = 0
            df.at[idx, 'rolling_team_reb_share'] = 0
            df.at[idx, 'rolling_pts_per_poss_fg'] = 0
            df.at[idx, 'rolling_opp_fta_rate'] = 0

        if (idx + 1) % 100 == 0:
            print(f"  Processed rolling derived features {idx + 1}/{len(df)} rows")

    # Rolling versions that require opponent data (will be calculated after opponent matching)
    df['rolling_quality_diff'] = 0
    df['rolling_off_vs_opp_def'] = 0
    df['rolling_def_vs_opp_off'] = 0

    # Remove each team's first game (where rolling stats are None/0)
    print("\nRemoving first game for each team (no prior stats)...")
    initial_count = len(df)
    df = df[df['GAME_INDEX'] > 0].reset_index(drop=True)
    removed_count = initial_count - len(df)
    print(f"  Removed {removed_count} first games ({removed_count//2} unique games)")

    # Get opponent rolling stats by matching with opponent's team ID
    print("\nMatching opponent rolling statistics...")

    # Get all rolling/nba/variance columns
    opp_rolling_cols = [col for col in df.columns if
                       col.startswith('rolling_') or
                       col.startswith('nba_') or
                       col.startswith('l3_') or
                       col.startswith('l5_') or
                       col.startswith('l10_') or
                       col == 'wins_last_10']

    for idx, row in df.iterrows():
        # Find opponent's row in the same game
        opp_row = df[(df['GAME_ID'] == row['GAME_ID']) & (df['TEAM_ID'] == row['OPP_TEAM_ID'])]

        if len(opp_row) > 0:
            for col in opp_rolling_cols:
                df.at[idx, f'opp_{col}'] = opp_row[col].values[0]

    # Now calculate matchup-based derived features (after opponent stats are available)
    print("\nCalculating matchup-based derived features...")

    for idx, row in df.iterrows():
        # Rolling matchup features
        if pd.notna(row['rolling_netrtg']) and pd.notna(row['opp_rolling_netrtg']):
            df.at[idx, 'rolling_quality_diff'] = row['rolling_netrtg'] - row['opp_rolling_netrtg']

        if pd.notna(row['rolling_ortg']) and pd.notna(row['opp_rolling_drtg']) and row['opp_rolling_drtg'] != 0:
            df.at[idx, 'rolling_off_vs_opp_def'] = row['rolling_ortg'] / row['opp_rolling_drtg']

        if pd.notna(row['rolling_drtg']) and pd.notna(row['opp_rolling_ortg']) and row['opp_rolling_ortg'] != 0:
            df.at[idx, 'rolling_def_vs_opp_off'] = row['rolling_drtg'] / row['opp_rolling_ortg']

        if (idx + 1) % 100 == 0:
            print(f"  Processed matchup features {idx + 1}/{len(df)} rows")

    # Last 3, 5, 10 games derived features
    for window in ['l3', 'l5', 'l10']:
        df[f'{window}_quality_diff'] = df[f'{window}_netrtg'] - df[f'opp_{window}_netrtg']
        df[f'{window}_off_vs_opp_def'] = df[f'{window}_ortg'] / df[f'opp_{window}_drtg'].replace(0, 1)
        df[f'{window}_def_vs_opp_off'] = df[f'{window}_drtg'] / df[f'opp_{window}_ortg'].replace(0, 1)
        df[f'{window}_combined_matchup_adv'] = (df[f'{window}_ortg'] - df[f'opp_{window}_drtg']) + (df[f'{window}_drtg'] - df[f'opp_{window}_ortg'])
        df[f'{window}_shooting_adv'] = df[f'{window}_efg'] - df[f'opp_{window}_efg']
        df[f'{window}_tov_adv'] = df[f'{window}_tov_pct'] - df[f'opp_{window}_tov_pct']
        df[f'{window}_oreb_adv'] = df[f'{window}_oreb_pct'] - df[f'opp_{window}_dreb_pct']

        # Calculate FTA rate advantage
        df[f'{window}_fta_rate'] = df[f'{window}_fta'] / df[f'{window}_fga'].replace(0, 1)
        df[f'opp_{window}_fta_rate'] = df[f'opp_{window}_opp_fta'] / df[f'opp_{window}_opp_fga'].replace(0, 1)
        df[f'{window}_fta_rate_adv'] = df[f'{window}_fta_rate'] - df[f'opp_{window}_fta_rate']

        # OPP shooting vs defense
        df[f'{window}_opp_shooting_vs_def'] = df[f'opp_{window}_efg'] * df[f'{window}_drtg']

    # Rolling versions of derived features (using simple rolling averages)
    df['rolling_combined_matchup_adv'] = (df['rolling_ortg'] - df['opp_rolling_drtg']) + (df['rolling_drtg'] - df['opp_rolling_ortg'])
    df['rolling_shooting_adv'] = df['rolling_efg'] - df['opp_rolling_efg']
    df['rolling_tov_adv'] = df['rolling_tov_pct'] - df['opp_rolling_tov_pct']
    df['rolling_oreb_adv'] = df['rolling_oreb_pct'] - df['opp_rolling_dreb_pct']

    # Calculate FTA rate for rolling
    print("\nCalculating FTA rate features...")
    for idx, row in df.iterrows():
        prev_games_df = df[(df['TEAM_ID'] == row['TEAM_ID']) & (df['GAME_INDEX'] < row['GAME_INDEX'])]
        if len(prev_games_df) > 0:
            total_fta = prev_games_df['FTA'].sum()
            total_fga = prev_games_df['FGA'].sum()
            df.at[idx, 'rolling_fta_rate'] = total_fta / total_fga if total_fga > 0 else 0
        else:
            df.at[idx, 'rolling_fta_rate'] = 0

        if (idx + 1) % 100 == 0:
            print(f"  Processed FTA rate features {idx + 1}/{len(df)} rows")

    df['rolling_fta_rate_adv'] = df['rolling_fta_rate'] - df['rolling_opp_fta_rate']

    # OPP shooting vs defense for rolling
    df['rolling_opp_shooting_vs_def'] = df['opp_rolling_efg'] * df['rolling_drtg']

    # Fill any remaining NaN values with 0
    df = df.fillna(0)

    # Final column cleanup and ordering
    print("\nOrganizing final columns...")

    # Generate all variance column names
    variance_stats = ['pts', 'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta',
                     'oreb', 'dreb', 'reb', 'ast', 'tov',
                     'fg_pct', 'fg3_pct', 'ft_pct',
                     'ortg', 'drtg', 'netrtg', 'pace',
                     'efg', 'ts', 'ast_pct', 'ast_to', 'ast_ratio',
                     'oreb_pct', 'dreb_pct', 'reb_pct', 'tov_pct',
                     'opp_netrtg', 'opp_ortg', 'opp_drtg',
                     'opp_efg', 'opp_tov_pct', 'opp_oreb_pct',
                     'opp_dreb_pct', 'opp_fta', 'opp_fga',
                     'opp_oreb', 'opp_dreb', 'poss',
                     'three_pt_dep_eff', 'two_pt_rate_eff',
                     'ft_to_fg_ratio', 'pct_pts_from_threes',
                     'oreb_battle_won', 'team_reb_share',
                     'pts_per_poss_fg']

    l3_cols = []
    l5_cols = []
    l10_cols = []

    for stat in variance_stats:
        l3_cols.extend([f'l3_{stat}', f'l3_{stat}_var'])
        l5_cols.extend([f'l5_{stat}', f'l5_{stat}_var'])
        l10_cols.extend([f'l10_{stat}', f'l10_{stat}_var'])

    opp_l3_cols = [f'opp_{col}' for col in l3_cols]
    opp_l5_cols = [f'opp_{col}' for col in l5_cols]
    opp_l10_cols = [f'opp_{col}' for col in l10_cols]

    final_columns = [
        'GAME_ID', 'SEASON', 'GAME_DATE', 'TEAM_ID', 'TEAM_NAME',
        'OPP_TEAM_ID', 'IS_HOME', 'WL',

        # Simple rolling averages (original features)
        'rolling_ortg', 'rolling_drtg', 'rolling_netrtg', 'rolling_pace',
        'rolling_ast_pct', 'rolling_ast_to', 'rolling_ast_ratio',
        'rolling_oreb_pct', 'rolling_dreb_pct', 'rolling_reb_pct',
        'rolling_tov_pct', 'rolling_efg', 'rolling_ts',
        'rolling_fg_pct', 'rolling_fg3_pct', 'rolling_ft_pct',
        'rolling_pts', 'rolling_starter_per', 'rolling_bench_per',
        'rolling_starter_usg', 'rolling_bench_usg',
        'wins_last_10', 'rolling_tempo', 'rolling_pts_diff',

        # Rolling derived features
        'rolling_quality_diff', 'rolling_off_vs_opp_def', 'rolling_def_vs_opp_off',
        'rolling_combined_matchup_adv', 'rolling_shooting_adv', 'rolling_tov_adv',
        'rolling_oreb_adv', 'rolling_three_pt_dep_eff', 'rolling_two_pt_rate_eff',
        'rolling_ft_to_fg_ratio', 'rolling_pct_pts_from_threes', 'rolling_oreb_battle_won',
        'rolling_team_reb_share', 'rolling_fta_rate', 'rolling_fta_rate_adv',
        'rolling_opp_fta_rate', 'rolling_opp_shooting_vs_def', 'rolling_pts_per_poss_fg',

        # NBA methodology (possession-weighted)
        'nba_ortg', 'nba_drtg', 'nba_netrtg', 'nba_pace',
        'nba_efg', 'nba_ts', 'nba_sefg', 'nba_ast_pct', 'nba_tov_pct',
        'nba_oreb_pct', 'nba_dreb_pct', 'nba_reb_pct',
    ]

    # Add L3, L5, L10 columns with variances
    final_columns.extend(l3_cols)

    # L3 derived features
    final_columns.extend([
        'l3_quality_diff', 'l3_off_vs_opp_def', 'l3_def_vs_opp_off',
        'l3_combined_matchup_adv', 'l3_shooting_adv', 'l3_tov_adv',
        'l3_oreb_adv', 'l3_fta_rate', 'l3_fta_rate_adv',
        'l3_opp_shooting_vs_def'
    ])

    final_columns.extend(l5_cols)

    # L5 derived features
    final_columns.extend([
        'l5_quality_diff', 'l5_off_vs_opp_def', 'l5_def_vs_opp_off',
        'l5_combined_matchup_adv', 'l5_shooting_adv', 'l5_tov_adv',
        'l5_oreb_adv', 'l5_fta_rate', 'l5_fta_rate_adv',
        'l5_opp_shooting_vs_def'
    ])

    final_columns.extend(l10_cols)

    # L10 derived features
    final_columns.extend([
        'l10_quality_diff', 'l10_off_vs_opp_def', 'l10_def_vs_opp_off',
        'l10_combined_matchup_adv', 'l10_shooting_adv', 'l10_tov_adv',
        'l10_oreb_adv', 'l10_fta_rate', 'l10_fta_rate_adv',
        'l10_opp_shooting_vs_def'
    ])

    # Opponent simple rolling averages
    final_columns.extend([
        'opp_rolling_ortg', 'opp_rolling_drtg', 'opp_rolling_netrtg', 'opp_rolling_pace',
        'opp_rolling_ast_pct', 'opp_rolling_ast_to', 'opp_rolling_ast_ratio',
        'opp_rolling_oreb_pct', 'opp_rolling_dreb_pct', 'opp_rolling_reb_pct',
        'opp_rolling_tov_pct', 'opp_rolling_efg', 'opp_rolling_ts',
        'opp_rolling_fg_pct', 'opp_rolling_fg3_pct', 'opp_rolling_ft_pct',
        'opp_rolling_pts', 'opp_rolling_starter_per', 'opp_rolling_bench_per',
        'opp_rolling_starter_usg', 'opp_rolling_bench_usg',
        'opp_wins_last_10', 'opp_rolling_tempo', 'opp_rolling_pts_diff',

        # Opponent rolling derived features
        'opp_rolling_quality_diff', 'opp_rolling_off_vs_opp_def', 'opp_rolling_def_vs_opp_off',
        'opp_rolling_combined_matchup_adv', 'opp_rolling_shooting_adv', 'opp_rolling_tov_adv',
        'opp_rolling_oreb_adv', 'opp_rolling_three_pt_dep_eff', 'opp_rolling_two_pt_rate_eff',
        'opp_rolling_ft_to_fg_ratio', 'opp_rolling_pct_pts_from_threes', 'opp_rolling_oreb_battle_won',
        'opp_rolling_team_reb_share', 'opp_rolling_fta_rate', 'opp_rolling_fta_rate_adv',
        'opp_rolling_opp_fta_rate', 'opp_rolling_opp_shooting_vs_def', 'opp_rolling_pts_per_poss_fg',

        # Opponent NBA methodology
        'opp_nba_ortg', 'opp_nba_drtg', 'opp_nba_netrtg', 'opp_nba_pace',
        'opp_nba_efg', 'opp_nba_ts', 'opp_nba_sefg', 'opp_nba_ast_pct', 'opp_nba_tov_pct',
        'opp_nba_oreb_pct', 'opp_nba_dreb_pct', 'opp_nba_reb_pct',
    ])

    # Add opponent L3, L5, L10 with variances and derived features
    final_columns.extend(opp_l3_cols)
    final_columns.extend([
        'opp_l3_quality_diff', 'opp_l3_off_vs_opp_def', 'opp_l3_def_vs_opp_off',
        'opp_l3_combined_matchup_adv', 'opp_l3_shooting_adv', 'opp_l3_tov_adv',
        'opp_l3_oreb_adv', 'opp_l3_fta_rate', 'opp_l3_fta_rate_adv',
        'opp_l3_opp_shooting_vs_def'
    ])

    final_columns.extend(opp_l5_cols)
    final_columns.extend([
        'opp_l5_quality_diff', 'opp_l5_off_vs_opp_def', 'opp_l5_def_vs_opp_off',
        'opp_l5_combined_matchup_adv', 'opp_l5_shooting_adv', 'opp_l5_tov_adv',
        'opp_l5_oreb_adv', 'opp_l5_fta_rate', 'opp_l5_fta_rate_adv',
        'opp_l5_opp_shooting_vs_def'
    ])

    final_columns.extend(opp_l10_cols)
    final_columns.extend([
        'opp_l10_quality_diff', 'opp_l10_off_vs_opp_def', 'opp_l10_def_vs_opp_off',
        'opp_l10_combined_matchup_adv', 'opp_l10_shooting_adv', 'opp_l10_tov_adv',
        'opp_l10_oreb_adv', 'opp_l10_fta_rate', 'opp_l10_fta_rate_adv',
        'opp_l10_opp_shooting_vs_def'
    ])

    available_cols = [col for col in final_columns if col in df.columns]
    df = df[available_cols]

    print(f"\n✓ Successfully processed {len(df)} game records")
    print(f"✓ Unique games: {df['GAME_ID'].nunique()}")
    print(f"✓ Teams: {df['TEAM_ID'].nunique()}")
    print(f"✓ Total features: {len(df.columns)}")
    print(f"✓ Variance features added for L3, L5, L10 windows")

    return df


def process_multiple_seasons(seasons, season_type='Regular Season', max_games_per_season=None):
    """
    Process multiple seasons and combine them.
    Parameters:
    -----------
    seasons : list
        List of seasons (e.g., ['2021-22', '2022-23', '2023-24'])
    season_type : str
        'Regular Season' or 'Playoffs'
    max_games_per_season : int, optional
        Limit games per season (for testing)
    Returns:
    --------
    pd.DataFrame
        Combined game data across all seasons
    """
    all_seasons = []

    for season in seasons:
        print("\n" + "="*80)
        print(f"PROCESSING SEASON: {season}")
        print("="*80)

        # Use season-specific progress files
        progress_id_file = f'nba_progress_{season.replace("-", "_")}.csv'
        progress_data_file = f'nba_game_data_progress_{season.replace("-", "_")}.csv'

        try:
            season_data = process_game_boxscores(
                season=season,
                season_type=season_type,
                max_games=max_games_per_season,
                progress_id_file=progress_id_file,
                progress_data_file=progress_data_file
            )
            all_seasons.append(season_data)
        except Exception as e:
            print(f"\n✗ Error processing {season}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if not all_seasons:
        raise Exception("Failed to process any season")

    combined = pd.concat(all_seasons, ignore_index=True)

    print("\n" + "="*80)
    print("COMBINED RESULTS")
    print("="*80)
    print(f"Total game records: {len(combined)}")
    print(f"Seasons: {combined['SEASON'].unique()}")
    print(f"Unique games: {combined['GAME_ID'].nunique()}")

    return combined


# Main execution
if __name__ == "__main__":

    print("="*80)
    print("NBA GAME-BY-GAME ROLLING STATISTICS BUILDER")
    print("WITH VARIANCE FEATURES (L3, L5, L10)")
    print("="*80)
    print("\n⚠ IMPORTANT: This script saves progress after each game")
    print("⚠ Progress is saved to TWO files:")
    print("⚠   1. nba_progress.csv - List of processed game IDs")
    print("⚠   2. nba_game_data_progress_2024_25.csv - Complete game data")
    print("⚠ If interrupted, it will resume from where it left off")
    print("⚠ NEW: Variance features for all L3, L5, L10 windows")
    print("⚠ NEW: No NaN values - uses 0 for first game and min_periods=1\n")

    try:
        # Process full season with progress saving
        df = process_game_boxscores(
            season='2024-25',  # change this to season to get data from
            season_type='Regular Season',
            progress_id_file='nba_progress_v3.csv',
            progress_data_file='nba_game_data_progress_2024_25v3.csv'  # change this to desired progress file
        )

        # Save final output
        output_file = 'nba_stats_2024_25_v3.csv'  # change this to desired output csv
        df.to_csv(output_file, index=False)
        print(f"\n✓ Final data saved to '{output_file}'")

        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"\nShape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())

        # Data quality check
        print("\n" + "="*80)
        print("DATA QUALITY CHECK")
        print("="*80)
        print(f"\nNull values per column:")
        null_counts = df.isnull().sum()
        if null_counts.sum() == 0:
            print("✓ No null values found!")
        else:
            print(null_counts[null_counts > 0].head(10))

        print("\n" + "="*80)
        print("NEW FEATURES SUMMARY")
        print("="*80)
        print("\n✓ Added variance features for L3, L5, L10 windows")
        print("✓ Added sEFG (stabilized effective field goal %)")
        print("✓ Added rolling point differential")
        print("✓ Points per 100 possessions already available as nba_ortg/rolling_ortg")
        print(f"\nVariance columns (sample):")
        var_cols = [col for col in df.columns if '_var' in col][:10]
        print(f"  {var_cols}")

        print("\n" + "="*80)
        print("TO PROCESS MULTIPLE SEASONS:")
        print("="*80)
        print("""
# Multiple seasons with resume capability:
seasons = ['2021-22', '2022-23', '2023-24']
df = process_multiple_seasons(seasons)
df.to_csv('nba_rolling_stats_multiple_seasons_with_variance.csv', index=False)

# To resume a specific season:
df = process_game_boxscores(
    season='2022-23',
    progress_id_file='nba_progress_2022_23.csv',
    progress_data_file='nba_game_data_progress_2022_23.csv'
)
        """)

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n⚠ Check the progress files - your data is saved!")
        print("⚠ Simply run the script again to resume from where it stopped")