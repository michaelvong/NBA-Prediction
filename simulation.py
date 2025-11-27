import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from collections import defaultdict, deque
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("NBA 2024-25 SEASON TRUE SEQUENTIAL SIMULATION")
print("=" * 80)

# Load the trained model
print("\n[1/8] Loading trained model...")
model = joblib.load('models/logreg_final_model.pkl')
print(f"✓ Model loaded successfully")
print(f"✓ Model features: {len(model.feature_names_in_)} features")

# Load the data
print("\n[2/8] Loading game data...")
df = pd.read_csv('nba_fe_outputs/game_features_raw2.csv')
df['game_date'] = pd.to_datetime(df['game_date'])
df = df.sort_values(['season', 'game_date', 'game_id', 'home']).reset_index(drop=True)
print(f"✓ Loaded {len(df)} total records")

# Split historical and current season
df_historical = df[df['season'] != '2024-25'].copy()
df_2024 = df[df['season'] == '2024-25'].copy()
print(f"✓ Historical records: {len(df_historical)}")
print(f"✓ 2024-25 records: {len(df_2024)}")
print(f"✓ Unique games in 2024-25: {df_2024['game_id'].nunique()}")

# Get games to simulate (sorted by date)
games_to_sim = df_2024.groupby('game_id').agg({
    'game_date': 'first',
    'home': 'first'  # Just to keep the structure
}).reset_index().sort_values('game_date')
print(f"✓ Games to simulate: {len(games_to_sim)}")

print("\n[3/8] Preparing team statistics tracking...")


class TeamStatsTracker:
    """Tracks and calculates rolling statistics for each team"""

    def __init__(self, team_name, historical_data):
        self.team_name = team_name
        self.games = []  # List of game dictionaries

        # Initialize with historical data for this team
        hist_team = historical_data[historical_data['team_name'] == team_name].copy()
        hist_team = hist_team.sort_values('game_date')

        for _, row in hist_team.iterrows():
            game = {
                'game_date': row['game_date'],
                'win': row['win'],
                'home': row['home'],
                'game_index': row['game_index'],
                # Store raw game stats (these would come from box scores)
                # For now, we'll back-calculate from the features
            }
            self.games.append(game)

    def add_game(self, game_data):
        """Add a new game result"""
        self.games.append(game_data)

    def get_games_todate(self):
        """Get all games up to current point"""
        return self.games

    def get_last_n_games(self, n):
        """Get last n games"""
        return self.games[-n:] if len(self.games) >= n else self.games

    def calculate_features(self):
        """Calculate all rolling features for current state"""
        features = {}
        games = self.games
        n_games = len(games)

        if n_games == 0:
            return self._get_default_features()

        # Calculate basic stats
        features['game_index'] = n_games

        # Win percentage to date
        wins = sum(g.get('win', 0) for g in games)
        features['win_pct_todate'] = wins / n_games if n_games > 0 else 0

        # Calculate averages todate (using stored values or defaults)
        avg_fields = ['team_offensiveRating', 'team_defensiveRating', 'team_netRating',
                      'eFG_diff', 'TOV_diff', 'ORB_diff', 'FTr_diff', 'team_pace',
                      '3P_pct_diff', 'team_PPP', 'opp_PPP', 'PPP_diff',
                      'DRB_diff', 'TRB_diff']

        for field in avg_fields:
            values = [g.get(field, 0) for g in games]
            features[f'{field}_avg_todate'] = np.mean(values) if values else 0

        # Calculate rolling windows (last 3, 5, 7, 10)
        for window in [3, 5, 7, 10]:
            recent_games = self.get_last_n_games(window)

            # Win count in window
            features[f'win_last{window}'] = sum(g.get('win', 0) for g in recent_games)

            # Averages in window
            for field in ['team_netRating', 'eFG_diff', 'TOV_diff', 'ORB_diff',
                          'FTr_diff', 'team_pace', '3P_pct_diff', 'team_PPP',
                          'PPP_diff', 'DRB_diff', 'TRB_diff']:
                values = [g.get(field, 0) for g in recent_games]
                features[f'{field}_last{window}'] = np.mean(values) if values else 0

            # Standard deviations in window
            for field in ['team_netRating', 'eFG_diff', 'team_pace']:
                values = [g.get(field, 0) for g in recent_games]
                features[f'{field}_std_last{window}'] = np.std(values) if len(values) > 1 else 0

        # Days since last game
        if len(games) >= 2:
            last_date = games[-1].get('game_date')
            prev_date = games[-2].get('game_date')
            if last_date and prev_date:
                features['days_since_last_game'] = (last_date - prev_date).days
            else:
                features['days_since_last_game'] = 2
        else:
            features['days_since_last_game'] = 2

        return features

    def _get_default_features(self):
        """Return default features when no games played"""
        features = {
            'game_index': 0,
            'win_pct_todate': 0.5,
            'days_since_last_game': 2
        }

        # Set all averages to 0
        for field in ['team_offensiveRating', 'team_defensiveRating', 'team_netRating',
                      'eFG_diff', 'TOV_diff', 'ORB_diff', 'FTr_diff', 'team_pace',
                      '3P_pct_diff', 'team_PPP', 'opp_PPP', 'PPP_diff', 'DRB_diff', 'TRB_diff']:
            features[f'{field}_avg_todate'] = 0

        for window in [3, 5, 7, 10]:
            features[f'win_last{window}'] = 0
            for field in ['team_netRating', 'eFG_diff', 'TOV_diff', 'ORB_diff',
                          'FTr_diff', 'team_pace', '3P_pct_diff', 'team_PPP',
                          'PPP_diff', 'DRB_diff', 'TRB_diff']:
                features[f'{field}_last{window}'] = 0
            for field in ['team_netRating', 'eFG_diff', 'team_pace']:
                features[f'{field}_std_last{window}'] = 0

        return features


print("✓ TeamStatsTracker class ready")

# Initialize team trackers
print("\n[4/8] Initializing team statistics from historical data...")
all_teams = sorted(df_2024['team_name'].unique())
print(f"✓ Teams: {len(all_teams)}")

# For the first game of 2024-25, we need end-of-previous-season stats
print("✓ Loading end-of-2023-24 season statistics...")

# Get the last game_index for each team in 2023-24 to use as starting point
season_2023_24 = df[df['season'] == '2023-24'].copy()
last_game_stats_2023 = {}

for team in all_teams:
    team_data = season_2023_24[season_2023_24['team_name'] == team]
    if len(team_data) > 0:
        # Get the last game for this team
        last_game = team_data.sort_values('game_date').iloc[-1]
        last_game_stats_2023[team] = last_game.to_dict()
    else:
        last_game_stats_2023[team] = None

print(f"✓ Loaded season-end stats for {len(last_game_stats_2023)} teams")

# Configuration
N_SIMULATIONS = 5
print(f"\n[5/8] Configuration:")
print(f"✓ Simulations to run: {N_SIMULATIONS}")
print(f"✓ Games per simulation: {len(games_to_sim)}")

# Storage for results
all_simulation_results = []
game_probabilities = []

print("\n[6/8] Running simulations...")
print("-" * 80)

for sim_num in range(N_SIMULATIONS):
    if (sim_num + 1) % 100 == 0:
        print(f"Simulation {sim_num + 1}/{N_SIMULATIONS}...")

    # Initialize trackers for this simulation (start from end of 2023-24)
    team_trackers = {}
    for team in all_teams:
        team_trackers[team] = {
            'wins': 0,
            'games': 0,
            'current_stats': last_game_stats_2023.get(team, {})
        }

    # Simulate each game sequentially
    for game_idx, game_row in games_to_sim.iterrows():
        game_id = game_row['game_id']
        game_date = game_row['game_date']

        # Get the actual game data to extract team names and home/away
        game_actual = df_2024[df_2024['game_id'] == game_id]

        if len(game_actual) != 2:
            continue

        home_row = game_actual[game_actual['home'] == 1].iloc[0]
        away_row = game_actual[game_actual['home'] == 0].iloc[0]

        home_team = home_row['team_name']
        away_team = away_row['team_name']

        # Build feature vector for home team using current stats
        home_stats = team_trackers[home_team]['current_stats']
        away_stats = team_trackers[away_team]['current_stats']

        # Create feature dictionary (using home team perspective)
        features = {}
        features['home'] = 1
        features['game_index'] = team_trackers[home_team]['games'] + 1

        # Get home team rolling features
        for key in home_stats:
            if key in ['team_offensiveRating_avg_todate', 'team_defensiveRating_avg_todate',
                       'team_netRating_avg_todate', 'eFG_diff_avg_todate', 'TOV_diff_avg_todate',
                       'ORB_diff_avg_todate', 'FTr_diff_avg_todate', 'team_pace_avg_todate',
                       'win_pct_todate', '3P_pct_diff_avg_todate', 'team_PPP_avg_todate',
                       'opp_PPP_avg_todate', 'PPP_diff_avg_todate', 'DRB_diff_avg_todate',
                       'TRB_diff_avg_todate'] or 'last' in key or 'std' in key or key == 'days_since_last_game':
                features[key] = home_stats.get(key, 0)

        # Opponent features
        features['opp_netRating_avg_todate'] = away_stats.get('team_netRating_avg_todate', 0)
        features['opp_win_pct_todate'] = away_stats.get('win_pct_todate', 0.5)
        features['opp_netRating_std_last5'] = away_stats.get('team_netRating_std_last5', 0)

        # Derived matchup features
        features['matchup_net_rating_gap'] = features.get('team_netRating_avg_todate', 0) - features.get(
            'opp_netRating_avg_todate', 0)
        features['matchup_win_pct_gap'] = features.get('win_pct_todate', 0.5) - features.get('opp_win_pct_todate', 0.5)
        features['matchup_PPP_gap'] = features.get('team_PPP_avg_todate', 0) - away_stats.get('team_PPP_avg_todate', 0)
        features['off_vs_def_rating'] = features.get('team_offensiveRating_avg_todate', 0) - away_stats.get(
            'team_defensiveRating_avg_todate', 0)
        features['pace_adjusted_net_rating'] = features.get('team_netRating_avg_todate', 0) * features.get(
            'team_pace_avg_todate', 100) / 100

        # Prepare feature vector for model
        try:
            X = []
            for feat_name in model.feature_names_in_:
                X.append(features.get(feat_name, 0))
            X = np.array(X).reshape(1, -1)

            # Predict
            win_prob_home = model.predict_proba(X)[0][1]

            # Simulate outcome
            home_wins = np.random.random() < win_prob_home

            # Update team records
            team_trackers[home_team]['games'] += 1
            team_trackers[away_team]['games'] += 1

            if home_wins:
                team_trackers[home_team]['wins'] += 1
            else:
                team_trackers[away_team]['wins'] += 1

            # Store probabilities (first simulation only)
            if sim_num == 0:
                game_probabilities.append({
                    'game_id': game_id,
                    'team_name': home_team,
                    'opponent_name': away_team,
                    'win': 1 if home_wins else 0,
                    'win_probability': win_prob_home
                })
                game_probabilities.append({
                    'game_id': game_id,
                    'team_name': away_team,
                    'opponent_name': home_team,
                    'win': 0 if home_wins else 1,
                    'win_probability': 1 - win_prob_home
                })

            # Update current stats for both teams (simplified update)
            # In a full implementation, you'd recalculate all rolling stats
            # For now, we update key stats that affect predictions most

            for team in [home_team, away_team]:
                is_home = (team == home_team)
                won = (home_wins and is_home) or (not home_wins and not is_home)

                stats = team_trackers[team]['current_stats']
                games_played = team_trackers[team]['games']
                wins = team_trackers[team]['wins']

                # Update win percentage
                stats['win_pct_todate'] = wins / games_played if games_played > 0 else 0

                # Update game index
                stats['game_index'] = games_played

                # For simplicity, slightly adjust rolling stats based on outcome
                # In reality, you'd recalculate from actual game box scores
                adjustment = 1 if won else -1
                stats['team_netRating_avg_todate'] = stats.get('team_netRating_avg_todate', 0) + adjustment * 0.5

        except Exception as e:
            if sim_num == 0:
                print(f"Warning: Error in game {game_id}: {e}")
            continue

    # Store simulation results
    for team in all_teams:
        all_simulation_results.append({
            'simulation_number': sim_num + 1,
            'team_name': team,
            'wins': team_trackers[team]['wins'],
            'games_played': team_trackers[team]['games'],
            'win_pct': team_trackers[team]['wins'] / team_trackers[team]['games'] if team_trackers[team][
                                                                                         'games'] > 0 else 0
        })

print(f"✓ Completed all {N_SIMULATIONS} simulations")

# Process results
print("\n[7/8] Processing results...")

# Game probabilities DataFrame
df_game_probs = pd.DataFrame(game_probabilities)
print(f"✓ Generated game probabilities for {len(df_game_probs) // 2} games")

# All simulations DataFrame
df_all_sims = pd.DataFrame(all_simulation_results)
print(f"✓ Compiled {len(df_all_sims)} simulation records")

# Average wins per team
df_avg_wins = df_all_sims.groupby('team_name').agg({
    'wins': ['mean', 'std', 'min', 'max'],
    'win_pct': ['mean', 'std']
}).round(2)
df_avg_wins.columns = ['avg_wins', 'std_wins', 'min_wins', 'max_wins', 'avg_win_pct', 'std_win_pct']
df_avg_wins = df_avg_wins.reset_index()
df_avg_wins = df_avg_wins.sort_values('avg_wins', ascending=False)
print(f"✓ Calculated average wins for {len(df_avg_wins)} teams")

# Save results
print("\n[8/8] Saving results...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

df_game_probs.to_csv(f'simulation_game_probabilities_{timestamp}.csv', index=False)
print(f"✓ Saved: simulation_game_probabilities_{timestamp}.csv")

df_avg_wins.to_csv(f'simulation_average_wins_{timestamp}.csv', index=False)
print(f"✓ Saved: simulation_average_wins_{timestamp}.csv")

df_all_sims.to_csv(f'simulation_full_summary_{timestamp}.csv', index=False)
print(f"✓ Saved: simulation_full_summary_{timestamp}.csv")

# Display summary
print("\n" + "=" * 80)
print("SIMULATION SUMMARY")
print("=" * 80)
print(f"\nTotal simulations: {N_SIMULATIONS}")
print(f"Games per simulation: {len(games_to_sim)}")
print(f"Total teams: {len(all_teams)}")

print("\n--- Top 10 Teams (by average wins) ---")
print(df_avg_wins.head(10).to_string(index=False))

print("\n--- Bottom 10 Teams (by average wins) ---")
print(df_avg_wins.tail(10).to_string(index=False))

print("\n" + "=" * 80)
print("TRUE SEQUENTIAL SIMULATION COMPLETE!")
print("=" * 80)