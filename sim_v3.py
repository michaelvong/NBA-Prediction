"""
simulate_season.py

Complete season simulation using:
 - Pretrained logistic regression (joblib)
 - Feature CSV that contains historical seasons and the 2024-25 schedule
 - Dynamic feature updating (realistic)
 - Monte Carlo simulation

Expectations about your CSV:
 - columns: 'game_id', 'season', 'team_name', 'opponent_team_name', 'home'
   where 'home' == 1 means the row is for the home team (i.e., this row is the home team's perspective)
"""

import os
import copy
import joblib
import random
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from tqdm import trange
import warnings

# -----------------------
# Ignore warnings
# -----------------------
warnings.filterwarnings("ignore")

# -----------------------
# User config
# -----------------------
MODEL_PATH = "models/logreg_final_model.pkl"
DATA_PATH = "nba_fe_outputs/game_features_raw2.csv"
OUTPUT_DIR = "simulation_results"
N_SIM = 20  # number of Monte Carlo simulations
RANDOM_SEED = 42
SEASON_TO_SIMULATE = "2024-25"

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -----------------------
# Feature columns (inference-time)
# Must match what model expects in training. Adjust ordering if needed.
# -----------------------
FEATURE_COLUMNS = [
    "home",
    "team_defensiveRating_avg_todate",
    "team_netRating_avg_todate",
    "opp_PPP_avg_todate",
    "PPP_diff_avg_todate",
    "DRB_diff_avg_todate",
    "team_netRating_last7",
    "eFG_diff_last5",
    "ORB_diff_last5",
    "TRB_diff_last5",
    "opp_netRating_avg_todate",
    "matchup_net_rating_gap",
    "matchup_win_pct_gap",
    "off_vs_def_rating",
    "pace_adjusted_net_rating",
    "home.1"
]

# -----------------------
# Helper classes / functions
# -----------------------

class TeamState:
    """
    Stores dynamic season state for one team during a simulation.
    """
    def __init__(self, team_name, preseason_baseline):
        self.team_name = team_name
        self.baseline = dict(preseason_baseline)

        # cumulative sums and counts for "avg_todate" features
        self.cum = defaultdict(float)
        self.count = defaultdict(int)

        # deques to track last-5 / last-7 per-stat
        self.last5 = defaultdict(lambda: deque(maxlen=5))
        self.last7 = defaultdict(lambda: deque(maxlen=7))

        # season results in this simulation
        self.wins = 0
        self.games_played = 0

    def record_game(self, stats_for_game, is_win):
        """
        Update team state with stats_for_game AFTER the game is played.
        """
        for k, v in stats_for_game.items():
            if v is None:
                continue
            self.cum[k] += v
            self.count[k] += 1

            if k in ("team_netRating", "eFG_diff", "ORB_diff", "TRB_diff"):
                if k == "team_netRating":
                    self.last7[k].append(v)
                else:
                    self.last5[k].append(v)

        self.games_played += 1
        if is_win:
            self.wins += 1

    def get_avg_todate(self, stat_key):
        if self.count.get(stat_key, 0) > 0:
            return self.cum[stat_key] / self.count[stat_key]
        else:
            return self.baseline.get(stat_key, self.baseline.get(stat_key.replace("_avg_todate", ""), 0.0))

    def get_lastN(self, stat_key, N):
        arr = []
        if N == 5:
            arr = list(self.last5.get(stat_key) or [])
        elif N == 7:
            arr = list(self.last7.get(stat_key) or [])
        if len(arr) > 0:
            return float(np.mean(arr))
        else:
            # fallback baseline for first game
            if stat_key == "team_netRating":
                return self.baseline.get("team_netRating_avg_todate", 0.0)
            elif stat_key == "eFG_diff":
                return self.baseline.get("eFG_diff_last5", 0.0)
            elif stat_key == "ORB_diff":
                return self.baseline.get("ORB_diff_last5", 0.0)
            elif stat_key == "TRB_diff":
                return self.baseline.get("TRB_diff_last5", 0.0)
            else:
                return 0.0

def load_model(path):
    return joblib.load(path)

def build_preseason_baselines(df, season_to_simulate):
    seasons_available = sorted(df["season"].unique())
    prev_seasons = [s for s in seasons_available if s < season_to_simulate]
    if len(prev_seasons) == 0:
        raise ValueError("No prior season data found.")
    last_season = prev_seasons[-1]

    prev = df[df["season"] == last_season]

    baseline_keys = {
        "team_defensiveRating_avg_todate": "team_defensiveRating_avg_todate",
        "team_netRating_avg_todate": "team_netRating_avg_todate",
        "opp_PPP_avg_todate": "opp_PPP_avg_todate",
        "PPP_diff_avg_todate": "PPP_diff_avg_todate",
        "DRB_diff_avg_todate": "DRB_diff_avg_todate",
        "team_netRating_last7": "team_netRating_last7",
        "eFG_diff_last5": "eFG_diff_last5",
        "ORB_diff_last5": "ORB_diff_last5",
        "TRB_diff_last5": "TRB_diff_last5",
        "opp_netRating_avg_todate": "opp_netRating_avg_todate",
        "pace_adjusted_net_rating": "pace_adjusted_net_rating",
        "off_vs_def_rating": "off_vs_def_rating",
        "elo": "elo"
    }

    team_baselines = {}
    teams = pd.concat([prev["team_name"], prev["opponent_team_name"]]).unique()
    for t in teams:
        trows = prev[prev["team_name"] == t]
        if len(trows) == 0:
            trows = prev[prev["opponent_team_name"] == t]

        base = {}
        for k, col in baseline_keys.items():
            if col in prev.columns:
                try:
                    val = float(trows[col].dropna().mean()) if len(trows) > 0 else np.nan
                except:
                    val = np.nan
                if np.isnan(val):
                    val = None
            else:
                val = None
            base[k] = val if val is not None else 0.0

        # last season win pct
        if len(trows) > 0 and "home_win" in trows.columns:
            try:
                wins = (trows["home_win"] == 1).sum()
                total = len(trows)
                win_pct = wins/total if total>0 else 0.5
            except:
                win_pct = 0.5
        else:
            win_pct = 0.5
        base["win_pct_last_season"] = win_pct

        team_baselines[t] = base

    return team_baselines

def assemble_feature_row(home_flag, team_state, opp_state):
    team_defensive = team_state.get_avg_todate("team_defensiveRating_avg_todate")
    team_net = team_state.get_lastN("team_netRating", 7)
    opp_ppp = opp_state.get_avg_todate("opp_PPP_avg_todate")
    ppp_diff = team_state.get_avg_todate("PPP_diff_avg_todate")
    drb_diff = team_state.get_avg_todate("DRB_diff_avg_todate")
    team_net_last7 = team_state.get_lastN("team_netRating", 7)
    efg_diff_last5 = team_state.get_lastN("eFG_diff", 5)
    orb_diff_last5 = team_state.get_lastN("ORB_diff", 5)
    trb_diff_last5 = team_state.get_lastN("TRB_diff", 5)
    opp_net = opp_state.get_avg_todate("team_netRating_avg_todate")
    matchup_net_gap = team_net - opp_net
    matchup_win_pct_gap = team_state.baseline.get("win_pct_last_season", 0.5) - opp_state.baseline.get("win_pct_last_season", 0.5)
    off_vs_def = team_state.baseline.get("off_vs_def_rating", 0.0)
    pace_adj = team_state.baseline.get("pace_adjusted_net_rating", 0.0)
    home_1 = home_flag

    row = [
        float(home_flag),
        float(team_defensive),
        float(team_net),
        float(opp_ppp),
        float(ppp_diff),
        float(drb_diff),
        float(team_net_last7),
        float(efg_diff_last5),
        float(orb_diff_last5),
        float(trb_diff_last5),
        float(opp_net),
        float(matchup_net_gap),
        float(matchup_win_pct_gap),
        float(off_vs_def),
        float(pace_adj),
        float(home_1)
    ]
    assert len(row) == len(FEATURE_COLUMNS)
    return np.array(row, dtype=float)

def make_observed(team_state, opp_state):
    """
    Generate observed stats for this game using previous simulated games
    """
    obs = {}

    obs["team_netRating"] = team_state.get_lastN("team_netRating", max(1, team_state.games_played)) + np.random.normal(scale=1.25)
    obs["eFG_diff"] = team_state.get_lastN("eFG_diff", max(1, team_state.games_played)) + np.random.normal(scale=0.005)
    obs["ORB_diff"] = team_state.get_lastN("ORB_diff", max(1, team_state.games_played)) + np.random.normal(scale=0.15)
    obs["TRB_diff"] = team_state.get_lastN("TRB_diff", max(1, team_state.games_played)) + np.random.normal(scale=0.15)
    obs["PPP_diff_avg_todate"] = team_state.get_avg_todate("PPP_diff_avg_todate") + np.random.normal(scale=0.01)
    obs["DRB_diff_avg_todate"] = team_state.get_avg_todate("DRB_diff_avg_todate") + np.random.normal(scale=0.15)
    obs["team_defensiveRating_avg_todate"] = team_state.get_avg_todate("team_defensiveRating_avg_todate") + np.random.normal(scale=0.1)
    obs["opp_netRating_avg_todate"] = opp_state.get_lastN("team_netRating", max(1, opp_state.games_played))
    return obs

def simulate_once(schedule_df, model, team_baselines, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    teams = sorted(list(set(pd.concat([schedule_df["team_name"], schedule_df["opponent_team_name"]]).unique())))
    team_states = {t: TeamState(t, team_baselines.get(t, {})) for t in teams}

    game_logs = []

    schedule_sorted = schedule_df.sort_values(by="game_id").reset_index(drop=True)

    for _, g in schedule_sorted.iterrows():
        home_team = g["team_name"]
        away_team = g["opponent_team_name"]
        home_flag = int(g.get("home", 1))

        home_state = team_states[home_team]
        away_state = team_states[away_team]

        feat_vec = assemble_feature_row(home_flag, home_state, away_state)
        proba = model.predict_proba(feat_vec.reshape(1, -1))[0,1]

        r = np.random.rand()
        home_wins = (r < proba)

        # observed stats from previous games / baseline
        home_obs = make_observed(home_state, away_state)
        away_obs = make_observed(away_state, home_state)

        home_state.record_game(home_obs, is_win=home_wins)
        away_state.record_game(away_obs, is_win=(not home_wins))

        game_logs.append({
            "game_id": g["game_id"],
            "home_team": home_team,
            "away_team": away_team,
            "prob_home_win": float(proba),
            "rand": float(r),
            "home_wins": bool(home_wins),
            "feature_vector": feat_vec.tolist()
        })

    wins = {t: team_states[t].wins for t in team_states}
    return wins, pd.DataFrame(game_logs)

# -----------------------
# Driver
# -----------------------
def main():
    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    required_cols = {"team_name", "opponent_team_name", "game_id", "season", "home"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    schedule_df = df[df["season"] == SEASON_TO_SIMULATE].copy()
    schedule_df = schedule_df[schedule_df["home"] == 1].drop_duplicates(subset=["game_id"])
    print(f"Number of games to simulate for season {SEASON_TO_SIMULATE}: {len(schedule_df)}")

    team_baselines = build_preseason_baselines(df, SEASON_TO_SIMULATE)

    print(f"Running {N_SIM} simulations...")
    all_sims_wins = []
    all_game_logs = []
    for s in trange(N_SIM):
        seed = RANDOM_SEED + s
        wins, game_log = simulate_once(schedule_df, model, team_baselines, seed=seed)
        all_sims_wins.append(wins)
        game_log["sim_id"] = s
        all_game_logs.append(game_log)

    wins_df = pd.DataFrame(all_sims_wins).reindex(sorted(team_baselines.keys()), axis=1)
    avg_wins = wins_df.mean(axis=0).sort_values(ascending=False)
    prob_make_playoffs = (wins_df >= 43).sum(axis=0) / N_SIM

    summary = pd.DataFrame({
        "team": avg_wins.index,
        "avg_wins": avg_wins.values,
        "prob_playoff_approx": prob_make_playoffs.reindex(avg_wins.index).values
    })

    print("Saving outputs...")
    summary.to_csv(os.path.join(OUTPUT_DIR, f"summary_{SEASON_TO_SIMULATE}.csv"), index=False)
    wins_df.to_csv(os.path.join(OUTPUT_DIR, f"all_sims_wins_{SEASON_TO_SIMULATE}.csv"), index=False)
    all_game_logs_df = pd.concat(all_game_logs, ignore_index=True)
    all_game_logs_df.to_csv(os.path.join(OUTPUT_DIR, f"all_game_logs_{SEASON_TO_SIMULATE}.csv"), index=False)

    print("Top teams by average simulated wins:")
    print(summary.head(15))
    print("Saved results in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
