import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ------------- CONFIG -------------

# TODO: update these paths to match your actual CSV filenames
SEASON_FILES = [
    "nba_team_stats_2021_22.csv",
    "nba_team_stats_2022_23.csv",
    "nba_team_stats_2023_24.csv",
    "nba_team_stats_2024_25.csv"
]

OUTPUT_DIR = "nba_fe_outputs"
CHECKPOINT_GAME = 20  # snapshot game index for single static features


# ------------- FEATURE ENGINEERING HELPERS -------------

def add_game_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "team_id", "game_date"]).copy()
    df["game_index"] = df.groupby(["season", "team_id"]).cumcount() + 1
    return df


def add_game_level_diffs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    return df


def add_season_to_date_features(df: pd.DataFrame) -> pd.DataFrame:
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
        "win",  # used for win_pct_todate
    ]

    for m in metrics:
        df[m + "_avg_todate"] = (
            df.groupby(group_cols)[m]
              .expanding()
              .mean()
              .shift(1)
              .reset_index(level=group_cols, drop=True)
        )

    # average of 0/1 win is season-to-date win percentage
    df["win_pct_todate"] = df["win_avg_todate"]
    return df


def add_rolling_features(df: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
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


def compute_team_season_net_rating(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["season", "team_id"], as_index=False)["team_netRating"].mean()
    grp = grp.rename(columns={"team_netRating": "team_netRating_season"})
    return grp


def add_sos_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    team_season = compute_team_season_net_rating(df)

    df = df.merge(
        team_season.rename(
            columns={
                "team_id": "opponent_team_id",
                "team_netRating_season": "opp_season_netRating",
            }
        ),
        on=["season", "opponent_team_id"],
        how="left",
    )

    df = df.sort_values(["season", "team_id", "game_date"])
    group_cols = ["season", "team_id"]

    df["opp_mean_netRating_todate"] = (
        df.groupby(group_cols)["opp_season_netRating"]
          .expanding()
          .mean()
          .shift(1)
          .reset_index(level=group_cols, drop=True)
    )

    df["opp_netRating_last10"] = (
        df.groupby(group_cols)["opp_season_netRating"]
          .rolling(window=10, min_periods=1)
          .mean()
          .shift(1)
          .reset_index(level=group_cols, drop=True)
    )

    df["adj_net_rating_todate"] = (
        df["team_netRating_avg_todate"] - df["opp_mean_netRating_todate"]
    )
    return df


def compute_home_away_summaries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby(["season", "team_id"])
    rows = []

    for (season, team_id), sub in g:
        total_games = len(sub)
        total_wins = sub["win"].sum()

        home_games = sub[sub["home"] == 1]
        away_games = sub[sub["home"] == 0]

        home_win_pct = home_games["win"].mean() if len(home_games) > 0 else np.nan
        away_win_pct = away_games["win"].mean() if len(away_games) > 0 else np.nan

        home_away_diff = (
            home_win_pct - away_win_pct
            if pd.notna(home_win_pct) and pd.notna(away_win_pct)
            else np.nan
        )

        final_win_pct = total_wins / total_games if total_games > 0 else np.nan

        rows.append(
            {
                "season": season,
                "team_id": team_id,
                "home_win_pct": home_win_pct,
                "away_win_pct": away_win_pct,
                "home_away_diff": home_away_diff,
                "final_win_pct": final_win_pct,
            }
        )

    return pd.DataFrame(rows)


def build_static_table(
    df: pd.DataFrame,
    home_away: pd.DataFrame,
    checkpoint_game: int = CHECKPOINT_GAME,
) -> pd.DataFrame:
    df_cp = df[df["game_index"] == checkpoint_game].copy()

    static = df_cp.merge(
        home_away, on=["season", "team_id"], how="left", suffixes=("", "_season")
    )

    feature_cols = [
        "team_offensiveRating_avg_todate",
        "team_defensiveRating_avg_todate",
        "team_netRating_avg_todate",
        "eFG_diff_avg_todate",
        "TOV_diff_avg_todate",
        "ORB_diff_avg_todate",
        "FTr_diff_avg_todate",
        "team_pace_avg_todate",
        "win_pct_todate",
        "opp_mean_netRating_todate",
        "adj_net_rating_todate",
        "team_netRating_last5",
        "team_netRating_last10",
        "eFG_diff_last5",
        "eFG_diff_last10",
        "win_last5",
        "win_last10",
        "team_netRating_std_last10",
        "eFG_diff_std_last10",
        "home_win_pct",
        "away_win_pct",
        "home_away_diff",
    ]

    target_col = "final_win_pct"

    static = static[["season", "team_id", "team_name"] + feature_cols + [target_col]]
    return static


def build_window_table(
    df: pd.DataFrame,
    home_away: pd.DataFrame,
    min_games: int = 10,
    step: int = 5,
) -> pd.DataFrame:
    """
    Build a static dataset with MANY rows per team-season, each one a 'window endpoint'
    e.g., game_index 10, 15, 20, ...
    """
    df = df.sort_values(["season", "team_id", "game_index"]).copy()

    ha_map = home_away.set_index(["season", "team_id"])["final_win_pct"].to_dict()

    feature_cols = [
        "team_offensiveRating_avg_todate",
        "team_defensiveRating_avg_todate",
        "team_netRating_avg_todate",
        "eFG_diff_avg_todate",
        "TOV_diff_avg_todate",
        "ORB_diff_avg_todate",
        "FTr_diff_avg_todate",
        "team_pace_avg_todate",
        "win_pct_todate",
        "opp_mean_netRating_todate",
        "adj_net_rating_todate",
        "team_netRating_last5",
        "team_netRating_last10",
        "eFG_diff_last5",
        "eFG_diff_last10",
        "win_last5",
        "win_last10",
        "team_netRating_std_last10",
        "eFG_diff_std_last10",
        "home_win_pct",
        "away_win_pct",
        "home_away_diff",
    ]

    df_ha = df.merge(home_away, on=["season", "team_id"], how="left")
    rows = []

    for (season, team_id), sub in df_ha.groupby(["season", "team_id"]):
        sub = sub.sort_values("game_index")
        max_game = sub["game_index"].max()
        endpoints = list(range(min_games, max_game + 1, step))

        for end_idx in endpoints:
            row = sub[sub["game_index"] == end_idx]
            if row.empty:
                continue
            row = row.iloc[0]

            out = {
                "season": season,
                "team_id": team_id,
                "team_name": row["team_name"],
                "game_index": end_idx,
                "final_win_pct": ha_map[(season, team_id)],
            }
            for c in feature_cols:
                out[c] = row[c]
            rows.append(out)

    window_df = pd.DataFrame(rows)
    return window_df


def build_sequences(
    df: pd.DataFrame,
    home_away: pd.DataFrame,
    seq_feature_cols=None,
):
    df = df.sort_values(["season", "team_id", "game_date"]).copy()

    if seq_feature_cols is None:
        seq_feature_cols = [
            "team_offensiveRating_avg_todate",
            "team_defensiveRating_avg_todate",
            "team_netRating_avg_todate",
            "eFG_diff_avg_todate",
            "TOV_diff_avg_todate",
            "ORB_diff_avg_todate",
            "FTr_diff_avg_todate",
            "team_pace_avg_todate",
            "win_pct_todate",
            "team_netRating_last5",
            "team_netRating_last10",
            "eFG_diff_last5",
            "eFG_diff_last10",
            "win_last5",
            "win_last10",
            "team_netRating_std_last10",
            "eFG_diff_std_last10",
            "opp_mean_netRating_todate",
            "adj_net_rating_todate",
            "home",
        ]

    df[seq_feature_cols] = df[seq_feature_cols].fillna(0.0)

    targets = (
        home_away.set_index(["season", "team_id"])["final_win_pct"].to_dict()
    )

    seasons = sorted(df["season"].unique())
    sequence_list = []
    target_list = []
    index_rows = []

    for season in seasons:
        teams = sorted(df[df["season"] == season]["team_id"].unique())
        for team_id in teams:
            sub = df[(df["season"] == season) & (df["team_id"] == team_id)]
            X_seq = sub[seq_feature_cols].to_numpy()
            sequence_list.append(X_seq)
            target_list.append(targets[(season, team_id)])
            team_name = sub["team_name"].iloc[0]
            index_rows.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "team_name": team_name,
                    "seq_index": len(sequence_list) - 1,
                }
            )

    max_len = max(seq.shape[0] for seq in sequence_list)
    num_features = sequence_list[0].shape[1]

    X = np.zeros((len(sequence_list), max_len, num_features), dtype=float)
    for i, seq in enumerate(sequence_list):
        X[i, : seq.shape[0], :] = seq

    y = np.array(target_list, dtype=float)
    index_df = pd.DataFrame(index_rows)

    return X, y, index_df, seq_feature_cols


def normalize_static(static: pd.DataFrame, feature_cols):
    scaler = StandardScaler()
    X = static[feature_cols].to_numpy()
    X_scaled = scaler.fit_transform(X)
    y = static["final_win_pct"].to_numpy()
    return X_scaled, y, scaler


def normalize_sequences(X: np.ndarray):
    N, T, F = X.shape
    X_2d = X.reshape(-1, F)
    X_norm = X.astype(float).copy()

    means = np.zeros(F)
    stds = np.ones(F)

    for j in range(F):
        col = X_2d[:, j]
        m = col.mean()
        s = col.std()
        if s == 0:
            s = 1.0
        means[j] = m
        stds[j] = s
        X_norm[:, :, j] = (X_norm[:, :, j] - m) / s

    return X_norm, means, stds


# ------------- MAIN -------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load and combine all season CSVs ----
    dfs = []
    for path in SEASON_FILES:
        df_season = pd.read_csv(path)
        dfs.append(df_season)

    df = pd.concat(dfs, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["season", "team_id", "game_date"])

    # ---- Feature pipeline ----
    df = add_game_level_diffs(df)
    df = add_game_index(df)
    df = add_season_to_date_features(df)
    df = add_rolling_features(df, windows=(5, 10))
    df = add_sos_features(df)
    home_away = compute_home_away_summaries(df)

    # ---- 1) Static snapshot per team-season ----
    static = build_static_table(df, home_away, checkpoint_game=CHECKPOINT_GAME)

    static_feature_cols = [
        c
        for c in static.columns
        if c not in ("season", "team_id", "team_name", "final_win_pct")
    ]

    X_static, y_static, scaler_static = normalize_static(static, static_feature_cols)

    static.to_csv(
        os.path.join(OUTPUT_DIR, "static_features_raw.csv"),
        index=False,
    )
    np.save(os.path.join(OUTPUT_DIR, "X_static.npy"), X_static)
    np.save(os.path.join(OUTPUT_DIR, "y_static.npy"), y_static)

    # ---- 2) Window-based static dataset ----
    window_static = build_window_table(
        df,
        home_away,
        min_games=10,
        step=5,
    )

    window_static.to_csv(
        os.path.join(OUTPUT_DIR, "static_features_windows_raw.csv"),
        index=False,
    )

    window_feature_cols = [
        c
        for c in window_static.columns
        if c not in ("season", "team_id", "team_name", "game_index", "final_win_pct")
    ]

    X_win = window_static[window_feature_cols].to_numpy()
    y_win = window_static["final_win_pct"].to_numpy()

    scaler_win = StandardScaler()
    X_win_scaled = scaler_win.fit_transform(X_win)

    np.save(os.path.join(OUTPUT_DIR, "X_static_windows.npy"), X_win_scaled)
    np.save(os.path.join(OUTPUT_DIR, "y_static_windows.npy"), y_win)

    # ---- 3) Sequences for RNN ----
    X_seq, y_seq, index_df, seq_feature_cols = build_sequences(df, home_away)
    X_seq_norm, seq_means, seq_stds = normalize_sequences(X_seq)

    np.save(os.path.join(OUTPUT_DIR, "X_seq_norm.npy"), X_seq_norm)
    np.save(os.path.join(OUTPUT_DIR, "y_seq.npy"), y_seq)
    index_df.to_csv(
        os.path.join(OUTPUT_DIR, "sequence_index_mapping.csv"),
        index=False,
    )

    # ---- 4) Save feature lists (optional) ----
    pd.Series(static_feature_cols).to_csv(
        os.path.join(OUTPUT_DIR, "static_feature_columns.txt"),
        index=False,
        header=False,
    )
    pd.Series(window_feature_cols).to_csv(
        os.path.join(OUTPUT_DIR, "static_window_feature_columns.txt"),
        index=False,
        header=False,
    )
    pd.Series(seq_feature_cols).to_csv(
        os.path.join(OUTPUT_DIR, "sequence_feature_columns.txt"),
        index=False,
        header=False,
    )

    print("Done. Files written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
