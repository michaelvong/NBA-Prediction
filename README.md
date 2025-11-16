CS271 ML project to predict outcome of nba game </br>
to generate data use run raw_script.py, edit season and output_file in main function


This will create an output directory named **`nba_fe_outputs`** containing all engineered features.

All season CSV files (e.g., `nba_team_stats_2021_22.csv`, `nba_team_stats_2022_23.csv`, etc.) are referenced directly inside **`build_nba_features.py`**.  
Update the file paths there if your data is stored elsewhere. Any data cleaning, column selection, or merging logic is also fully defined within that script.

---

## Feature Engineering Overview

The goal of this pipeline is to transform raw *per-team per-game* NBA statistics into a rich, multi-season feature set that supports:

- **Regression models** (predict exact final win%)
- **Classification models** (predict performance tiers)
- **Sequence models** (RNNs that read game-by-game trajectories)

To achieve this, the script performs:

---

### 1. Combine All Seasons

All CSVs are loaded and concatenated. Rows are sorted by `season`, `team_id`, and `game_date`.  
This creates one unified dataset covering multiple years.

---

### 2. Compute Game-Level Differential Features

For every game, the script creates **team minus opponent** features such as:

- Net points and net rating  
- Offensive/defensive rating gaps  
- eFG%, TS%, FTr, TOV% differentials  
- Pace and possession differences  

Differential stats capture *relative strength* and are highly predictive.

---

### 3. Rolling Performance Windows

The pipeline generates rolling features for each team:

- Last **5-game** and **10-game** averages  
- Rolling shooting efficiency  
- Rolling turnover and rebound trends  
- Rolling net rating and pace trends  

Rolling windows smooth randomness and capture momentum.

---

### 4. Season-to-Date Summaries

For each game index, the script computes cumulative statistics:

- Season average offensive/defensive efficiency  
- Season net rating  
- Season shooting and turnover tendencies  
- Aggregate possession and pace metrics  

These represent long-term team strength.

---

### 5. Strength of Schedule (SOS)

Using opponents’ stats, the script computes:

- Opponent win% difficulty  
- Rolling opponent quality  
- SOS-adjusted efficiency metrics  

This compensates for uneven schedules across teams.

---

### 6. Home/Away Splits

Teams often perform differently at home vs. away.  
We compute:

- Home-only averages  
- Away-only averages  
- Home–away differentials  

---

### 7. Create Model-Ready Feature Tables

#### **A) Static Snapshot Dataset — One Row per Team-Season**
Pulled at a fixed checkpoint (e.g., after 20 games).  
Used for regression or simple classification.

Outputs:
- `static_features_raw.csv`
- `X_static.npy`
- `y_static.npy`

#### **B) Windowed Dataset — Multiple Samples per Season**
A new sample every *N* games (e.g., every 5).  
Useful for probability models and midseason forecasting.

Outputs:
- `static_features_windows_raw.csv`
- `X_static_windows.npy`
- `y_static_windows.npy`

#### **C) Sequence Dataset — Full Game-by-Game Timeline**
Used for RNNs.  
Each sequence contains all game-level features in chronological order.

Outputs:
- `X_seq_norm.npy`
- `y_seq.npy`
- `sequence_index_mapping.csv`

---

### 8. Standardization & Feature Selection

All feature arrays are standardized (z-score).  
Feature column lists are exported for reproducibility:

- `static_feature_columns.txt`
- `static_window_feature_columns.txt`
- `sequence_feature_columns.txt`

---

## Summary

Running `build_nba_features.py` automatically transforms raw NBA game data into:

- **Static per-season features**
- **Rolling window features**
- **Sequential game-by-game features**

These support:

- Logistic Regression  
- XGBoost / Random Forest  
- RNNs for season trajectory modeling  

All preprocessing, windowing, SOS, and differential feature logic is handled inside the script.
