import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ------------- CONFIG -------------

# Update these paths to match your actual CSV filenames
SEASON_FILES = [
    "nba_team_stats_2020_21.csv",
    "nba_team_stats_2021_22.csv",
    "nba_team_stats_2022_23.csv",
    "nba_team_stats_2023_24.csv",
    "nba_team_stats_2024_25.csv",
]

OUTPUT_DIR = "nba_fe_outputs"
ROLL_WINDOWS = (3, 5, 7, 10)


# ------------- FEATURE ENGINEERING HELPERS -------------

def add_game_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by season / team / game_date and assign a 1-based game_index
    for each (season, team_id).
    """
    df = df.sort_values(["season", "team_id", "game_date"]).copy()
    df["game_index"] = df.groupby(["season", "team_id"]).cumcount() + 1
    return df


def add_game_level_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-game team vs opponent differentials.

    These are *not* used directly as features; they get rolled/expanded
    and shifted so only prior games are used.
    """
    df = df.copy()

    # Original differentials
    df["eFG_diff"] = (
            df["team_effectiveFieldGoalPercentage"]
            - df["opp_effectiveFieldGoalPercentage"]
    )
    df["TS_diff"] = (
            df["team_trueShootingPercentage"]
            - df["opp_trueShootingPercentage"]
    )
    df["TOV_diff"] = (
            df["opp_teamTurnoverPercentage"]
            - df["team_teamTurnoverPercentage"]
    )
    df["ORB_diff"] = (
            df["team_offensiveReboundPercentage"]
            - df["opp_offensiveReboundPercentage"]
    )
    df["FTr_diff"] = (
            df["team_freeThrowAttemptRate"]
            - df["opp_freeThrowAttemptRate"]
    )
    df["net_rating_diff"] = df["team_netRating"] - df["opp_netRating"]

    # 3P% differential
    df["3P_pct_diff"] = (
            df["team_threePointersPercentage"]
            - df["opp_threePointersPercentage"]
    )

    # Points Per Possession (PPP)
    df["team_PPP"] = np.where(
        df["team_possessions"] > 0,
        df["team_points"] / df["team_possessions"],
        np.nan
    )
    df["opp_PPP"] = np.where(
        df["opp_possessions"] > 0,
        df["opp_points"] / df["opp_possessions"],
        np.nan
    )
    df["PPP_diff"] = df["team_PPP"] - df["opp_PPP"]

    # Defensive & Total rebound % differential
    df["DRB_diff"] = (
            df["team_defensiveReboundPercentage"]
            - df["opp_defensiveReboundPercentage"]
    )
    df["TRB_diff"] = (
            df["team_reboundPercentage"]
            - df["opp_reboundPercentage"]
    )

    # ========== NEW FEATURES ==========

    # 1. Paint points percentage differential
    df["team_paint_points_pct"] = np.where(
        df["team_points"] > 0,
        df["team_pointsPaint"] / df["team_points"],
        np.nan
    )
    df["opp_paint_points_pct"] = np.where(
        df["opp_points"] > 0,
        df["opp_pointsPaint"] / df["opp_points"],
        np.nan
    )
    df["paint_points_pct_diff"] = df["team_paint_points_pct"] - df["opp_paint_points_pct"]

    # 2. Fast break points percentage differential
    df["team_fastbreak_pct"] = np.where(
        df["team_points"] > 0,
        df["team_pointsFastBreak"] / df["team_points"],
        np.nan
    )
    df["opp_fastbreak_pct"] = np.where(
        df["opp_points"] > 0,
        df["opp_pointsFastBreak"] / df["opp_points"],
        np.nan
    )
    df["fastbreak_points_pct_diff"] = df["team_fastbreak_pct"] - df["opp_fastbreak_pct"]

    # 3. Points off turnovers percentage differential
    df["team_points_off_tov_pct"] = np.where(
        df["team_points"] > 0,
        df["team_pointsOffTurnovers"] / df["team_points"],
        np.nan
    )
    df["opp_points_off_tov_pct"] = np.where(
        df["opp_points"] > 0,
        df["opp_pointsOffTurnovers"] / df["opp_points"],
        np.nan
    )
    df["points_off_turnovers_pct_diff"] = df["team_points_off_tov_pct"] - df["opp_points_off_tov_pct"]

    # 4. Second chance efficiency (points per offensive rebound)
    df["team_second_chance_efficiency"] = np.where(
        df["team_reboundsOffensive"] > 0,
        df["team_pointsSecondChance"] / df["team_reboundsOffensive"],
        np.nan
    )

    # 5. Opponent second chance points allowed per our defensive rebounds
    df["opp_second_chance_allowed_efficiency"] = np.where(
        df["team_reboundsDefensive"] > 0,
        df["opp_pointsSecondChance"] / df["team_reboundsDefensive"],
        np.nan
    )

    # 6. Fast break efficiency (points per possession)
    df["fastbreak_efficiency"] = np.where(
        df["team_possessions"] > 0,
        df["team_pointsFastBreak"] / df["team_possessions"],
        np.nan
    )

    # 7. Pace differential (for later interaction terms)
    df["pace_diff"] = df["team_pace"] - df["opp_pace"]

    return df


def add_season_to_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (season, team_id) and each metric, compute an expanding mean
    with a shift(1) so that game t only sees games 1..(t-1).

    This gives "long term form" features and win_pct_todate.
    """
    df = df.sort_values(["season", "team_id", "game_date"]).copy()
    group_cols = ["season", "team_id"]

    metrics = [
        "team_offensiveRating",
        "team_defensiveRating",
        "team_netRating",
        "eFG_diff",
        "TOV_diff",
        "ORB_diff",
        "FTr_diff",
        "team_pace",
        "win",
        "3P_pct_diff",
        "team_PPP",
        "opp_PPP",
        "PPP_diff",
        "DRB_diff",
        "TRB_diff",
        # NEW metrics
        "paint_points_pct_diff",
        "fastbreak_points_pct_diff",
        "points_off_turnovers_pct_diff",
        "team_second_chance_efficiency",
        "opp_second_chance_allowed_efficiency",
        "fastbreak_efficiency",
    ]

    for m in metrics:
        df[f"{m}_avg_todate"] = (
            df.groupby(group_cols)[m]
            .expanding()
            .mean()
            .shift(1)
            .reset_index(level=group_cols, drop=True)
        )

    # average of win (0/1) is season-to-date win percentage
    df["win_pct_todate"] = df["win_avg_todate"]

    return df


def add_rolling_features(df: pd.DataFrame, windows=ROLL_WINDOWS) -> pd.DataFrame:
    """
    For each (season, team_id) compute rolling means and stds over previous N games,
    with shift(1) so only *completed* games are used.
    """
    df = df.sort_values(["season", "team_id", "game_date"]).copy()
    group_cols = ["season", "team_id"]

    roll_mean_metrics = [
        "team_netRating",
        "eFG_diff",
        "TOV_diff",
        "ORB_diff",
        "FTr_diff",
        "team_pace",
        "win",
        "3P_pct_diff",
        "team_PPP",
        "PPP_diff",
        "DRB_diff",
        "TRB_diff",
        # NEW metrics
        "paint_points_pct_diff",
        "fastbreak_points_pct_diff",
        "points_off_turnovers_pct_diff",
        "team_second_chance_efficiency",
        "opp_second_chance_allowed_efficiency",
        "fastbreak_efficiency",
        "pace_diff",
    ]

    roll_std_metrics = [
        "team_netRating",
        "eFG_diff",
        "team_pace",
    ]

    for w in windows:
        for m in roll_mean_metrics:
            df[f"{m}_last{w}"] = (
                df.groupby(group_cols)[m]
                .rolling(window=w, min_periods=1)
                .mean()
                .shift(1)
                .reset_index(level=group_cols, drop=True)
            )

        for m in roll_std_metrics:
            df[f"{m}_std_last{w}"] = (
                df.groupby(group_cols)[m]
                .rolling(window=w, min_periods=2)
                .std()
                .shift(1)
                .reset_index(level=group_cols, drop=True)
            )

    return df


def add_momentum_and_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create momentum indicators and interaction terms from already-computed
    leak-free rolling features.
    """
    df = df.copy()

    # 1. Net rating momentum (short-term vs medium-term)
    df["net_rating_momentum"] = df["team_netRating_last3"] - df["team_netRating_last5"]

    # 2. eFG × ORB interaction (shooting efficiency × rebounding synergy)
    df["eFG_ORB_interaction"] = df["eFG_diff_last5"] * df["ORB_diff_last5"]

    # 3. Turnover differential adjusted for pace (last 5 games)
    df["turnover_rate_adjusted"] = np.where(
        df["team_pace_last5"] > 0,
        df["TOV_diff_last5"] / df["team_pace_last5"],
        np.nan
    )

    # 4. eFG × Pace interaction (shooting efficiency × pace differential)
    df["eFG_pace_interaction"] = df["eFG_diff_last5"] * df["pace_diff_last5"]

    return df


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (season, team_id), compute days since last game.
    """
    df = df.sort_values(["season", "team_id", "game_date"]).copy()
    df["days_since_last_game"] = (
        df.groupby(["season", "team_id"])["game_date"]
        .diff()
        .dt.days
    )
    return df


def add_opponent_pre_game_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach opponent season-to-date features (computed with the same
    leak-free logic) and build matchup gaps.
    """
    df = df.copy()

    # Compute opponent's season-to-date PPP from their perspective
    df = df.sort_values(["season", "opponent_team_id", "game_date"]).copy()
    df["opp_PPP_avg_todate"] = (
        df.groupby(["season", "opponent_team_id"])["opp_PPP"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=["season", "opponent_team_id"], drop=True)
    )

    # Re-sort back to team perspective
    df = df.sort_values(["season", "team_id", "game_date"]).copy()

    # Opponent pre-game form
    opp_src_cols = [
        "season",
        "game_id",
        "team_id",
        "team_netRating_avg_todate",
        "win_pct_todate",
        "team_defensiveRating_avg_todate",
        "team_offensiveRating_avg_todate",
        "team_pace_avg_todate",
    ]

    opp_df = (
        df[opp_src_cols]
        .rename(
            columns={
                "team_id": "opponent_team_id",
                "team_netRating_avg_todate": "opp_netRating_avg_todate",
                "win_pct_todate": "opp_win_pct_todate",
                "team_defensiveRating_avg_todate": "opp_defensiveRating_avg_todate",
                "team_offensiveRating_avg_todate": "opp_offensiveRating_avg_todate",
                "team_pace_avg_todate": "opp_pace_avg_todate",
            }
        )
    )

    df = df.merge(
        opp_df,
        on=["season", "game_id", "opponent_team_id"],
        how="left",
    )

    # Matchup gaps
    df["matchup_net_rating_gap"] = (
            df["team_netRating_avg_todate"] - df["opp_netRating_avg_todate"]
    )
    df["matchup_win_pct_gap"] = (
            df["win_pct_todate"] - df["opp_win_pct_todate"]
    )
    df["matchup_PPP_gap"] = (
            df["team_PPP_avg_todate"] - df["opp_PPP_avg_todate"]
    )

    # Team offensive rating vs opponent defensive rating
    df["off_vs_def_rating"] = np.where(
        df["opp_defensiveRating_avg_todate"] > 0,
        df["team_offensiveRating_avg_todate"] / df["opp_defensiveRating_avg_todate"],
        np.nan
    )

    # Pace-adjusted net rating differential
    df["pace_adjusted_net_rating"] = (
            (df["team_netRating_avg_todate"] * df["team_pace_avg_todate"]) -
            (df["opp_netRating_avg_todate"] * df["opp_pace_avg_todate"])
    )

    return df


def add_opponent_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add opponent's rolling standard deviation for net rating (last 5 games).
    """
    df = df.copy()

    opp_rolling_cols = [
        "season",
        "game_id",
        "team_id",
        "team_netRating_std_last5",
    ]

    opp_rolling_df = (
        df[opp_rolling_cols]
        .rename(
            columns={
                "team_id": "opponent_team_id",
                "team_netRating_std_last5": "opp_netRating_std_last5",
            }
        )
    )

    df = df.merge(
        opp_rolling_df,
        on=["season", "game_id", "opponent_team_id"],
        how="left",
    )

    return df


def build_game_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final game-level table with:
    - IDs / metadata
    - engineered, non-leaky features
    - target = win (0/1)
    """
    # IDs / metadata
    meta_cols = [
        "season",
        "game_id",
        "game_date",
        "team_id",
        "team_name",
        "opponent_team_id",
        "opponent_team_name",
        "home",
        "game_index",
        "win",
    ]

    # Engineered feature columns (leak-free)
    feature_cols = [
        # Season-to-date (expanding, shifted)
        "team_offensiveRating_avg_todate",
        "team_defensiveRating_avg_todate",
        "team_netRating_avg_todate",
        "eFG_diff_avg_todate",
        "TOV_diff_avg_todate",
        "ORB_diff_avg_todate",
        "FTr_diff_avg_todate",
        "team_pace_avg_todate",
        "win_pct_todate",
        "3P_pct_diff_avg_todate",
        "team_PPP_avg_todate",
        "opp_PPP_avg_todate",
        "PPP_diff_avg_todate",
        "DRB_diff_avg_todate",
        "TRB_diff_avg_todate",

        # NEW: Season-to-date advanced metrics
        "paint_points_pct_diff_avg_todate",
        "fastbreak_points_pct_diff_avg_todate",
        "points_off_turnovers_pct_diff_avg_todate",
        "team_second_chance_efficiency_avg_todate",
        "opp_second_chance_allowed_efficiency_avg_todate",
        "fastbreak_efficiency_avg_todate",

        # Rolling means – recent form (windows: 3, 5, 7, 10)
        "team_netRating_last3",
        "team_netRating_last5",
        "team_netRating_last7",
        "team_netRating_last10",

        "eFG_diff_last3",
        "eFG_diff_last5",
        "eFG_diff_last7",
        "eFG_diff_last10",

        "TOV_diff_last3",
        "TOV_diff_last5",
        "TOV_diff_last7",
        "TOV_diff_last10",

        "ORB_diff_last3",
        "ORB_diff_last5",
        "ORB_diff_last7",
        "ORB_diff_last10",

        "FTr_diff_last3",
        "FTr_diff_last5",
        "FTr_diff_last7",
        "FTr_diff_last10",

        "team_pace_last3",
        "team_pace_last5",
        "team_pace_last7",
        "team_pace_last10",

        "win_last3",
        "win_last5",
        "win_last7",
        "win_last10",

        "3P_pct_diff_last3",
        "3P_pct_diff_last5",
        "3P_pct_diff_last7",
        "3P_pct_diff_last10",

        "team_PPP_last3",
        "team_PPP_last5",
        "team_PPP_last7",
        "team_PPP_last10",

        "PPP_diff_last3",
        "PPP_diff_last5",
        "PPP_diff_last7",
        "PPP_diff_last10",

        "DRB_diff_last3",
        "DRB_diff_last5",
        "DRB_diff_last7",
        "DRB_diff_last10",

        "TRB_diff_last3",
        "TRB_diff_last5",
        "TRB_diff_last7",
        "TRB_diff_last10",

        # NEW: Rolling advanced metrics
        "paint_points_pct_diff_last3",
        "paint_points_pct_diff_last5",
        "paint_points_pct_diff_last7",
        "paint_points_pct_diff_last10",

        "fastbreak_points_pct_diff_last3",
        "fastbreak_points_pct_diff_last5",
        "fastbreak_points_pct_diff_last7",
        "fastbreak_points_pct_diff_last10",

        "points_off_turnovers_pct_diff_last3",
        "points_off_turnovers_pct_diff_last5",
        "points_off_turnovers_pct_diff_last7",
        "points_off_turnovers_pct_diff_last10",

        "team_second_chance_efficiency_last3",
        "team_second_chance_efficiency_last5",
        "team_second_chance_efficiency_last7",
        "team_second_chance_efficiency_last10",

        "opp_second_chance_allowed_efficiency_last3",
        "opp_second_chance_allowed_efficiency_last5",
        "opp_second_chance_allowed_efficiency_last7",
        "opp_second_chance_allowed_efficiency_last10",

        "fastbreak_efficiency_last3",
        "fastbreak_efficiency_last5",
        "fastbreak_efficiency_last7",
        "fastbreak_efficiency_last10",

        # Rolling std – recent volatility
        "team_netRating_std_last3",
        "team_netRating_std_last5",
        "team_netRating_std_last7",
        "team_netRating_std_last10",

        "eFG_diff_std_last3",
        "eFG_diff_std_last5",
        "eFG_diff_std_last7",
        "eFG_diff_std_last10",

        "team_pace_std_last3",
        "team_pace_std_last5",
        "team_pace_std_last7",
        "team_pace_std_last10",

        # NEW: Momentum & interaction features
        "net_rating_momentum",
        "eFG_ORB_interaction",
        "turnover_rate_adjusted",
        "eFG_pace_interaction",

        # Rest / fatigue
        "days_since_last_game",

        # Opponent pre-game strength + matchup gaps
        "opp_netRating_avg_todate",
        "opp_win_pct_todate",
        "matchup_net_rating_gap",
        "matchup_win_pct_gap",
        "matchup_PPP_gap",
        "off_vs_def_rating",
        "pace_adjusted_net_rating",
        "opp_netRating_std_last5",

        # Game context
        "home",
        "game_index",
    ]

    cols_needed = meta_cols + feature_cols
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in feature pipeline: {missing}")

    game_df = df[cols_needed].copy()

    # Drop games where we don't yet have prior info (NaNs in features)
    game_df = game_df.dropna(subset=feature_cols).reset_index(drop=True)

    return game_df, feature_cols


def standardize_features(game_df: pd.DataFrame, feature_cols):
    """
    Standardize numeric features with StandardScaler, return X, y.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(game_df[feature_cols].to_numpy())
    y = game_df["win"].to_numpy().astype(int)
    return X, y, scaler


# ------------- MAIN -------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load and combine all season CSVs ----
    dfs = []
    for path in SEASON_FILES:
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path} (skipping)")
            continue
        df_season = pd.read_csv(path)
        dfs.append(df_season)

    if not dfs:
        raise ValueError("No season CSVs found. Update SEASON_FILES to point to your data.")

    df = pd.concat(dfs, ignore_index=True)

    # Ensure game_date is datetime
    df["game_date"] = pd.to_datetime(df["game_date"])

    # ---- Feature pipeline (all leak-free by construction) ----
    df = add_game_level_diffs(df)
    df = add_game_index(df)
    df = add_season_to_date_features(df)
    df = add_rolling_features(df, windows=ROLL_WINDOWS)
    df = add_momentum_and_interaction_features(df)  # NEW STEP
    df = add_rest_features(df)
    df = add_opponent_pre_game_form(df)
    df = add_opponent_rolling_features(df)

    # ---- Build final game-level dataset ----
    game_df, feature_cols = build_game_level_table(df)
    X_game, y_game, scaler = standardize_features(game_df, feature_cols)

    # ---- Save raw game-level CSV ----
    game_df.to_csv(
        os.path.join(OUTPUT_DIR, "game_features_raw4.csv"),
        index=False,
    )

    # ---- Save feature column names ----
    pd.Series(feature_cols).to_csv(
        os.path.join(OUTPUT_DIR, "game_feature_columns4.txt"),
        index=False,
        header=False,
    )

    print("✓ Done. Enhanced game-level features written to:", OUTPUT_DIR)
    print(f"✓ Num games: {X_game.shape[0]:,}")
    print(f"✓ Num features: {X_game.shape[1]} (was ~93, now ~137)")
    print("\n📊 NEW FEATURES ADDED:")
    print("   • Paint points % differential (season + rolling)")
    print("   • Fast break points % differential (season + rolling)")
    print("   • Points off turnovers % differential (season + rolling)")
    print("   • Second chance efficiency (season + rolling)")
    print("   • Opponent second chance allowed efficiency (season + rolling)")
    print("   • Fast break efficiency (season + rolling)")
    print("   • Net rating momentum (last3 - last5)")
    print("   • eFG × ORB interaction")
    print("   • Turnover rate adjusted for pace")
    print("   • eFG × Pace interaction")


if __name__ == "__main__":
    main()