CS271 ML project to predict outcome of nba game </br>
to generate data use run raw_script.py, edit season and output_file in main function


# Game-Level Feature Engineering Overview

The goal of this pipeline is to transform raw *per-team per-game* NBA data into a fully leak-free feature set that supports:

- **Game-level win prediction models** (XGBoost, Logistic Regression)
- **Opponent-aware modeling** (pre-game matchup strength)
- **Sequence models** (RNNs / LSTMs reading team trajectories game-by-game)

This new version is fully redesigned for **per-game** prediction, ensuring:

- **No label leakage** (features use only games *before* the current game)
- **Time-aware features** (chronological sorting, game index)
- **Opponent pre-game strength** (their own season-to-date form)
- **Short-term form and volatility** (rolling windows)
- **Rest/fatigue modeling** (days since last game)

---

### 1. Input CSV Structure (Raw Stats)

Each seasonal input file (e.g., `nba_team_stats_2020_21.csv`) contains **one row per team per game** with:

- Identifiers: `season`, `game_id`, `game_date`, `team_id`, `team_name`, `opponent_team_id`, `home`, `win`
- Team advanced stats: offensive/defensive/net rating, eFG%, TS%, TOV%, ORB%, FTr, pace  
- Opponent advanced stats: same fields prefixed with `opp_`

These are **current-game stats** collected after the game, so they **cannot** be used directly as features.

---

### 2. Chronological Ordering & Game Index

All games are sorted by: season → team_id → game_date


A per-team `game_index` is assigned:

- `1` = first game of season  
- Increases sequentially for each team  

This gives models awareness of **season progression**, letting them learn differences between early-season and late-season performance.

---

### 3. Convert Raw Stats to Game-Level Differentials

The script computes internal **team minus opponent** metrics:

- `eFG_diff`, `TS_diff`, `TOV_diff`, `ORB_diff`, `FTr_diff`
- `net_rating_diff`

These reflect performance gaps but are used **only** inside rolling/expanding windows.  
They are **never fed directly** into the model (to avoid leakage).

---

### 4. Season-to-Date Features (Shifted = Leak-Free)

For each team and season, the script computes expanding means of:

- `team_offensiveRating`, `team_defensiveRating`, `team_netRating`
- `eFG_diff`, `TOV_diff`, `ORB_diff`, `FTr_diff`
- `team_pace`
- `win` → becomes `win_pct_todate`

All are **shifted by 1 game**:

> For game *t*, features summarize games **1..(t–1)** only.

This produces long-term form features that are **100% pre-game**.

---

### 5. Rolling Windows (Short-Term Form & Volatility)

Using 5-game and 10-game windows, the script computes leak-free rolling:

- Means: `team_netRating_last5`, `win_last10`, `eFG_diff_last10`
- Standard deviations: performance volatility, e.g.  
  `team_netRating_std_last10`, `eFG_diff_std_last5`

All rolling calculations are **shifted** so they contain only prior games.

These features capture:

- Momentum  
- Slumps  
- Consistency vs volatility  

---

### 6. Rest / Fatigue Metrics

For each team:

days_since_last_game = difference in days from previous game


This encodes:

- Back-to-backs  
- Long rest periods  
- Fatigue effects  

---

### 7. Opponent Pre-Game Strength

For every game, the opponent’s **own** season-to-date features are merged in:

- `opp_netRating_avg_todate`
- `opp_win_pct_todate`

Then matchup feature gaps are computed:

- `matchup_net_rating_gap`
- `matchup_win_pct_gap`

This lets the model compare teams based on **pre-game strength**, not final-season stats.

---

### 8. Output Datasets

After running the script:

- **`game_features_raw.csv`** — full leak-free dataset  
- **`X_game.npy`** — standardized numerical features  
- **`y_game.npy`** — per-team game outcome (0/1)  
- **`game_feature_columns.txt`** — names of input features for modeling  

These outputs are directly compatible with:
- Logistic Regression
- XGBoost / Random Forest
- Neural networks
- RNN/LSTM team-sequence modeling

---

### Summary

This pipeline converts raw NBA per-game stats into a **robust, leak-free dataset** suitable for game-level prediction.  
It includes temporal structure, opponent context, short-term form, long-term form, rest effects, and matchup strength — all computed using **only information available before each game**.



