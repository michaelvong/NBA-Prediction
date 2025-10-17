"""
NBA Game-by-Game Rolling Statistics Builder
Fetches game-by-game data with rolling team and player statistics.
Each game has TWO rows - one for each team with opponent stats.
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder, boxscoreadvancedv2, boxscoretraditionalv2


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


def calculate_rolling_stats(game_data, team_id, game_index):
    """
    Calculate rolling statistics up to (but not including) the current game.
    Includes both simple averages and NBA's possession-weighted methodology.

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
        # First game of season - return None/default values
        return {
            # Simple averages (existing)
            'rolling_ortg': None,
            'rolling_drtg': None,
            'rolling_netrtg': None,
            'rolling_pace': None,
            'rolling_efg': None,
            'rolling_ts': None,
            'rolling_ast_pct': None,
            'rolling_ast_to': None,
            'rolling_ast_ratio': None,
            'rolling_oreb_pct': None,
            'rolling_dreb_pct': None,
            'rolling_reb_pct': None,
            'rolling_tov_pct': None,
            'rolling_fg_pct': None,
            'rolling_fg3_pct': None,
            'rolling_ft_pct': None,
            'rolling_pts': None,
            'rolling_starter_per': None,
            'rolling_bench_per': None,
            'rolling_starter_usg': None,
            'rolling_bench_usg': None,
            'wins_last_10': 0,

            # NBA methodology (possession-weighted)
            'nba_ortg': None,
            'nba_drtg': None,
            'nba_netrtg': None,
            'nba_pace': None,
            'nba_efg': None,
            'nba_ts': None,
            'nba_ast_pct': None,
            'nba_tov_pct': None,
            'nba_oreb_pct': None,
            'nba_dreb_pct': None,
            'nba_reb_pct': None,
        }

    # Calculate simple rolling averages (existing)
    stats = {
        'rolling_ortg': prev_games['OFF_RATING'].mean() if 'OFF_RATING' in prev_games else None,
        'rolling_drtg': prev_games['DEF_RATING'].mean() if 'DEF_RATING' in prev_games else None,
        'rolling_netrtg': prev_games['NET_RATING'].mean() if 'NET_RATING' in prev_games else None,
        'rolling_pace': prev_games['PACE'].mean() if 'PACE' in prev_games else None,
        'rolling_efg': prev_games['EFG_PCT'].mean() if 'EFG_PCT' in prev_games else None,
        'rolling_ts': prev_games['TS_PCT'].mean() if 'TS_PCT' in prev_games else None,
        'rolling_ast_pct': prev_games['AST_PCT'].mean() if 'AST_PCT' in prev_games else None,
        'rolling_ast_to': prev_games['AST_TO'].mean() if 'AST_TO' in prev_games else None,
        'rolling_ast_ratio': prev_games['AST_RATIO'].mean() if 'AST_RATIO' in prev_games else None,
        'rolling_oreb_pct': prev_games['OREB_PCT'].mean() if 'OREB_PCT' in prev_games else None,
        'rolling_dreb_pct': prev_games['DREB_PCT'].mean() if 'DREB_PCT' in prev_games else None,
        'rolling_reb_pct': prev_games['REB_PCT'].mean() if 'REB_PCT' in prev_games else None,
        'rolling_tov_pct': prev_games['TOV_PCT'].mean() if 'TOV_PCT' in prev_games else None,
        'rolling_fg_pct': prev_games['FG_PCT'].mean() if 'FG_PCT' in prev_games else None,
        'rolling_fg3_pct': prev_games['FG3_PCT'].mean() if 'FG3_PCT' in prev_games else None,
        'rolling_ft_pct': prev_games['FT_PCT'].mean() if 'FT_PCT' in prev_games else None,
        'rolling_pts': prev_games['PTS'].mean() if 'PTS' in prev_games else None,
        'rolling_starter_per': prev_games['STARTER_PER_AVG'].mean() if 'STARTER_PER_AVG' in prev_games else None,
        'rolling_bench_per': prev_games['BENCH_PER_AVG'].mean() if 'BENCH_PER_AVG' in prev_games else None,
        'rolling_starter_usg': prev_games['STARTER_USG_AVG'].mean() if 'STARTER_USG_AVG' in prev_games else None,
        'rolling_bench_usg': prev_games['BENCH_USG_AVG'].mean() if 'BENCH_USG_AVG' in prev_games else None,
    }

    # Wins in last 10 games
    last_10 = prev_games.tail(10)
    stats['wins_last_10'] = last_10['WL'].apply(lambda x: 1 if x == 'W' else 0).sum()

    # NBA Methodology - Possession-weighted cumulative stats
    # These calculations match NBA.com's official methodology

    # Get cumulative totals
    total_poss = prev_games['POSS'].sum() if 'POSS' in prev_games.columns else 0
    total_pts = prev_games['PTS'].sum() if 'PTS' in prev_games.columns else 0
    total_opp_pts = prev_games['OPP_PTS'].sum() if 'OPP_PTS' in prev_games.columns else 0

    # Offensive Rating = Points per 100 possessions
    stats['nba_ortg'] = (total_pts / total_poss * 100) if total_poss > 0 else None

    # Defensive Rating = Opponent points per 100 possessions
    stats['nba_drtg'] = (total_opp_pts / total_poss * 100) if total_poss > 0 else None

    # Net Rating = ORTG - DRTG
    if stats['nba_ortg'] is not None and stats['nba_drtg'] is not None:
        stats['nba_netrtg'] = stats['nba_ortg'] - stats['nba_drtg']
    else:
        stats['nba_netrtg'] = None

    # Pace = Possessions per 48 minutes (weighted by minutes played)
    total_min = prev_games['MIN_PLAYED'].sum() if 'MIN_PLAYED' in prev_games.columns else 0
    stats['nba_pace'] = (total_poss / total_min * 48) if total_min > 0 else None

    # EFG% = (FGM + 0.5 * 3PM) / FGA
    total_fgm = prev_games['FGM'].sum() if 'FGM' in prev_games.columns else 0
    total_fg3m = prev_games['FG3M'].sum() if 'FG3M' in prev_games.columns else 0
    total_fga = prev_games['FGA'].sum() if 'FGA' in prev_games.columns else 0
    stats['nba_efg'] = ((total_fgm + 0.5 * total_fg3m) / total_fga) if total_fga > 0 else None

    # TS% = PTS / (2 * (FGA + 0.44 * FTA))
    total_fta = prev_games['FTA'].sum() if 'FTA' in prev_games.columns else 0
    ts_denominator = 2 * (total_fga + 0.44 * total_fta)
    stats['nba_ts'] = (total_pts / ts_denominator) if ts_denominator > 0 else None

    # AST% = AST / FGM (team assists per made field goal)
    total_ast = prev_games['AST'].sum() if 'AST' in prev_games.columns else 0
    stats['nba_ast_pct'] = (total_ast / total_fgm * 100) if total_fgm > 0 else None

    # TOV% = TOV / (FGA + 0.44 * FTA + TOV)
    total_tov = prev_games['TOV'].sum() if 'TOV' in prev_games.columns else 0
    tov_denominator = total_fga + 0.44 * total_fta + total_tov
    stats['nba_tov_pct'] = (total_tov / tov_denominator * 100) if tov_denominator > 0 else None

    # OREB% = OREB / (OREB + OPP_DREB)
    total_oreb = prev_games['OREB'].sum() if 'OREB' in prev_games.columns else 0
    total_opp_dreb = prev_games['OPP_DREB'].sum() if 'OPP_DREB' in prev_games.columns else 0
    oreb_denominator = total_oreb + total_opp_dreb
    stats['nba_oreb_pct'] = (total_oreb / oreb_denominator * 100) if oreb_denominator > 0 else None

    # DREB% = DREB / (DREB + OPP_OREB)
    total_dreb = prev_games['DREB'].sum() if 'DREB' in prev_games.columns else 0
    total_opp_oreb = prev_games['OPP_OREB'].sum() if 'OPP_OREB' in prev_games.columns else 0
    dreb_denominator = total_dreb + total_opp_oreb
    stats['nba_dreb_pct'] = (total_dreb / dreb_denominator * 100) if dreb_denominator > 0 else None

    # REB% = (OREB + DREB) / (OREB + DREB + OPP_OREB + OPP_DREB)
    total_reb = total_oreb + total_dreb
    total_opp_reb = total_opp_oreb + total_opp_dreb
    reb_denominator = total_reb + total_opp_reb
    stats['nba_reb_pct'] = (total_reb / reb_denominator * 100) if reb_denominator > 0 else None

    return stats


def process_game_boxscores(season='2023-24', season_type='Regular Season', max_games=None):
    """
    Process all games in a season to create game-by-game rolling statistics.

    Parameters:
    -----------
    season : str
        Season in format 'YYYY-YY'
    season_type : str
        'Regular Season' or 'Playoffs'
    max_games : int, optional
        Limit number of games to process (for testing)

    Returns:
    --------
    pd.DataFrame
        Game-by-game data with rolling statistics
    """
    # Get all games
    games = fetch_games_for_season(season, season_type)

    # Get unique game IDs
    unique_games = games['GAME_ID'].unique()

    if max_games:
        unique_games = unique_games[:max_games]
        print(f"\n⚠ Processing only first {max_games} games for testing")

    print(f"\nProcessing {len(unique_games)} games...")
    print("This will take a while due to API rate limits...\n")

    all_game_data = []

    for idx, game_id in enumerate(unique_games):
        print(f"Processing game {idx+1}/{len(unique_games)}: {game_id}")

        # Get game info from games dataframe
        game_info = games[games['GAME_ID'] == game_id]

        if len(game_info) != 2:
            print(f"  ⚠ Skipping game {game_id} - incomplete data")
            continue

        # Fetch box scores
        player_trad, team_trad = fetch_box_score_traditional(game_id)
        time.sleep(0.6)  # Rate limiting

        player_adv, team_adv = fetch_box_score_advanced(game_id)
        time.sleep(0.6)  # Rate limiting

        if player_trad is None or player_adv is None or team_trad is None or team_adv is None:
            print(f"  ⚠ Skipping game {game_id} - box score unavailable")
            continue

        # Process each team in the game
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
            starter_per_avg = starters['PIE'].mean() if len(starters) > 0 else None
            bench_per_avg = bench['PIE'].mean() if len(bench) > 0 else None
            starter_usg_avg = starters['USG_PCT'].mean() if len(starters) > 0 else None
            bench_usg_avg = bench['USG_PCT'].mean() if len(bench) > 0 else None

            # Create game record with safe column access
            game_record = {
                'GAME_ID': game_id,
                'SEASON': season,
                'GAME_DATE': team_game['GAME_DATE'],
                'MATCHUP': team_game['MATCHUP'],
                'TEAM_ID': team_id,
                'TEAM_NAME': team_game['TEAM_NAME'],
                'OPP_TEAM_ID': opp_id,
                'IS_HOME': 1 if 'vs.' in team_game['MATCHUP'] else 0,
                'WL': team_game['WL'],

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
            }

            all_game_data.append(game_record)

    # Create DataFrame
    df = pd.DataFrame(all_game_data)

    # Add game index for rolling calculations
    df = df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
    # FIX: Explicitly convert GAME_INDEX to int to avoid string comparison errors
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

    # Remove each team's first game (where rolling stats are None)
    print("\nRemoving first game for each team (no prior stats)...")
    initial_count = len(df)
    df = df[df['GAME_INDEX'] > 0].reset_index(drop=True)
    removed_count = initial_count - len(df)
    print(f"  Removed {removed_count} first games ({removed_count//2} unique games)")

    # Get opponent rolling stats by matching with opponent's team ID
    print("\nMatching opponent rolling statistics...")
    opp_rolling_cols = [col for col in df.columns if col.startswith('rolling_') or col.startswith('nba_') or col == 'wins_last_10']

    for idx, row in df.iterrows():
        # Find opponent's row in the same game
        opp_row = df[(df['GAME_ID'] == row['GAME_ID']) & (df['TEAM_ID'] == row['OPP_TEAM_ID'])]

        if len(opp_row) > 0:
            for col in opp_rolling_cols:
                df.at[idx, f'opp_{col}'] = opp_row[col].values[0]

    # Final column cleanup and ordering
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
        'wins_last_10', 'rolling_tempo',

        # NBA methodology (possession-weighted cumulative)
        'nba_ortg', 'nba_drtg', 'nba_netrtg', 'nba_pace',
        'nba_efg', 'nba_ts', 'nba_ast_pct', 'nba_tov_pct',
        'nba_oreb_pct', 'nba_dreb_pct', 'nba_reb_pct',

        # Opponent simple rolling averages
        'opp_rolling_ortg', 'opp_rolling_drtg', 'opp_rolling_netrtg', 'opp_rolling_pace',
        'opp_rolling_ast_pct', 'opp_rolling_ast_to', 'opp_rolling_ast_ratio',
        'opp_rolling_oreb_pct', 'opp_rolling_dreb_pct', 'opp_rolling_reb_pct',
        'opp_rolling_tov_pct', 'opp_rolling_efg', 'opp_rolling_ts',
        'opp_rolling_fg_pct', 'opp_rolling_fg3_pct', 'opp_rolling_ft_pct',
        'opp_rolling_pts', 'opp_rolling_starter_per', 'opp_rolling_bench_per',
        'opp_rolling_starter_usg', 'opp_rolling_bench_usg',
        'opp_wins_last_10', 'opp_rolling_tempo',

        # Opponent NBA methodology
        'opp_nba_ortg', 'opp_nba_drtg', 'opp_nba_netrtg', 'opp_nba_pace',
        'opp_nba_efg', 'opp_nba_ts', 'opp_nba_ast_pct', 'opp_nba_tov_pct',
        'opp_nba_oreb_pct', 'opp_nba_dreb_pct', 'opp_nba_reb_pct',
    ]

    available_cols = [col for col in final_columns if col in df.columns]
    df = df[available_cols]

    print(f"\n✓ Successfully processed {len(df)} game records")
    print(f"✓ Unique games: {df['GAME_ID'].nunique()}")
    print(f"✓ Teams: {df['TEAM_ID'].nunique()}")

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

        try:
            season_data = process_game_boxscores(
                season=season,
                season_type=season_type,
                max_games=max_games_per_season
            )
            all_seasons.append(season_data)
        except Exception as e:
            print(f"\n✗ Error processing {season}: {str(e)}")
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

    # IMPORTANT: Start with a small test first!
    print("="*80)
    print("NBA GAME-BY-GAME ROLLING STATISTICS BUILDER")
    print("="*80)
    print("\n⚠ WARNING: Processing full seasons takes several hours due to API rate limits")
    print("⚠ Starting with 50 games for testing (to see rolling stats develop)...\n")

    try:
        # Test with 50 games to see rolling stats in action
        test_df = process_game_boxscores(
            season='2023-24',
            season_type='Regular Season',
            max_games=50
        )

        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"\nShape: {test_df.shape}")
        print(f"\nColumns: {test_df.columns.tolist()}")

        # Show data quality check
        print("\n" + "="*80)
        print("DATA QUALITY CHECK")
        print("="*80)

        # Check how many games each team has
        games_per_team = test_df.groupby('TEAM_NAME').size()
        print(f"\nGames per team:")
        print(games_per_team.head(10))

        # Check rolling stats - should have values after first game
        print(f"\nRolling stats NULL count (should decrease as season progresses):")
        rolling_cols = [col for col in test_df.columns if col.startswith('rolling_')]
        for col in rolling_cols[:5]:  # Show first 5
            null_count = test_df[col].isna().sum()
            print(f"  {col}: {null_count}/{len(test_df)} null ({100*null_count/len(test_df):.1f}%)")

        # Show example of team with multiple games
        if len(games_per_team) > 0:
            team_with_most_games = games_per_team.idxmax()
            print(f"\n{team_with_most_games} progression (showing rolling stats develop):")
            team_data = test_df[test_df['TEAM_NAME'] == team_with_most_games]

            # Select columns that exist
            display_cols = ['GAME_DATE', 'WL']
            optional_cols = ['MATCHUP', 'rolling_ortg', 'rolling_drtg', 'rolling_netrtg', 'rolling_pts']
            for col in optional_cols:
                if col in team_data.columns:
                    display_cols.append(col)

            team_sample = team_data[display_cols].head(5)
            print(team_sample.to_string())

        print(f"\nFirst few rows:")
        print(test_df.head())

        # Save test results
        test_df.to_csv('nba_rolling_stats_TEST.csv', index=False)
        print("\n✓ Test data saved to 'nba_rolling_stats_TEST.csv'")

        print("\n" + "="*80)
        print("TO PROCESS FULL SEASONS:")
        print("="*80)
        print("""
# Single full season (takes ~2-3 hours):
df = process_game_boxscores(season='2023-24', season_type='Regular Season')
df.to_csv('nba_rolling_stats_2023_24.csv', index=False)

# Multiple seasons:
seasons = ['2021-22', '2022-23', '2023-24']
df = process_multiple_seasons(seasons)
df.to_csv('nba_rolling_stats_multiple_seasons.csv', index=False)
        """)

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
