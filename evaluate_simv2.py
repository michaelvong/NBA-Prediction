import pandas as pd
import numpy as np
from sklearn.metrics import (
    log_loss, 
    brier_score_loss, 
    roc_auc_score, 
    accuracy_score
)
from scipy.stats import pearsonr

# ============================
# FILE PATHS
# ============================
GAME_LOG_FILE = "simulation_results/all_game_logs_2024-25.csv"
SEASON_SIM_FILE = "simulation_results/summary_2024-25.csv"
ACTUAL_WINS_FILE = "season_wins.txt"   # your text file of real standings


# ============================
# HELPER: Parse actual W-L into dict
# ============================
import re

def parse_actual_standings(path):
    text = open(path).read()
    pattern = r"([A-Za-z0-9\s]+):\s*(\d+)-(\d+)"
    matches = re.findall(pattern, text)

    team_to_wins = {}
    for team, wins, losses in matches:
        team_clean = " ".join(team.split()).strip()
        team_to_wins[team_clean] = int(wins)

    return pd.Series(team_to_wins, name="actual_wins")


# ============================
# MAIN
# ============================

def main():

    # ======================================
    # 1. PER-GAME PROBABILITY EVALUATION
    # ======================================
    print("\n=== LOADING GAME LOGS ===")
    df = pd.read_csv(GAME_LOG_FILE)

    df["p_pred"] = df["prob_home_win"].astype(float)
    df["y_true"] = df["home_wins"].astype(int)

    # --- Probabilistic Metrics ---
    ll = log_loss(df["y_true"], df["p_pred"])
    bs = brier_score_loss(df["y_true"], df["p_pred"])

    try:
        auc = roc_auc_score(df["y_true"], df["p_pred"])
    except ValueError:
        auc = None

    acc = accuracy_score(df["y_true"], (df["p_pred"] >= 0.5).astype(int))

    # ======================================
    # 2. SEASON-LEVEL EVALUATION
    # ======================================
    print("\n=== LOADING SEASON RESULTS ===")

    sim_df = pd.read_csv(SEASON_SIM_FILE)
    actual_series = parse_actual_standings(ACTUAL_WINS_FILE)

    # Normalize names to match (strip spaces)
    sim_df["team"] = sim_df["team"].astype(str).str.strip()
    actual_series.index = actual_series.index.str.strip()

    merged = sim_df.merge(actual_series, left_on="team", right_index=True, how="left")

    if merged["actual_wins"].isna().any():
        missing = merged[merged["actual_wins"].isna()]["team"].tolist()
        raise ValueError(f"Missing actual win totals for: {missing}")

    merged["error"] = merged["avg_wins"] - merged["actual_wins"]
    merged["abs_error"] = merged["error"].abs()
    merged["sq_error"] = merged["error"] ** 2

    mae = merged["abs_error"].mean()
    rmse = np.sqrt(merged["sq_error"].mean())
    corr, pval = pearsonr(merged["avg_wins"], merged["actual_wins"])

    # ======================================
    # 3. PRINT RESULTS
    # ======================================

    print("\n====================== PER-GAME METRICS ======================")
    print(f"Log Loss:       {ll:.5f}")
    print(f"Brier Score:    {bs:.5f}")
    print(f"ROC AUC:        {auc:.5f}" if auc is not None else "ROC AUC:        Not available")
    print(f"Accuracy @0.5:  {acc:.5f}")

    # Calibration histogram (10 bins)
    print("\nCalibration Bins (avg predicted vs actual outcomes):")
    bins = np.linspace(0, 1, 11)
    df["bin"] = pd.cut(df["p_pred"], bins)

    calib = df.groupby("bin").agg(
        avg_pred=("p_pred", "mean"),
        actual_rate=("y_true", "mean"),
        count=("y_true", "size")
    )
    print(calib)

    print("\n==================== SEASON-LEVEL METRICS ====================")
    print(f"MAE (Wins):     {mae:.2f}")
    print(f"RMSE (Wins):    {rmse:.2f}")
    print(f"Correlation:    {corr:.4f}")
    print(f"P-value:        {pval:.4g}")
    print("==============================================================")

    print("\nTop Overestimates:")
    print(merged.sort_values("error", ascending=False).head(5)[["team", "avg_wins", "actual_wins", "error"]])

    print("\nTop Underestimates:")
    print(merged.sort_values("error").head(5)[["team", "avg_wins", "actual_wins", "error"]])

    print("\nDone!\n")


if __name__ == "__main__":
    main()
