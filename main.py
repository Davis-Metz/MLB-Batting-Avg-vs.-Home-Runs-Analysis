"""
Are MLB teams more successful with a higher team batting average or more home runs?

Setup (run once in a terminal):
    pip install pandas scipy statsmodels matplotlib

Data: Teams.csv from the Lahman database (https://sabr.org/lahman-database/).
Column names below (yearID, teamID, W, AB, H, HR) may need small changes if
your version of the file differs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats

# ---------------------------------------------------------------- 1. Load data
# pybaseball's built-in Lahman download is broken, so load Teams.csv yourself:
#   1. Go to https://sabr.org/lahman-database/ and download the latest CSV zip.
#   2. Unzip it and find Teams.csv (it may be inside a subfolder).
#   3. Put Teams.csv in your PyCharm project folder, next to main.py,
#      or change the path below to wherever the file is.
TEAMS_CSV = "Teams.csv"
teams = pd.read_csv(TEAMS_CSV)    # one row per team-season

# ---------------------------------------------------------------- 2. Clean
SEASONS = [2018, 2019, 2021, 2022, 2023]   # 2020 excluded (60-game season)

df = teams[teams["yearID"].isin(SEASONS)].copy()
df = df.rename(columns={"yearID": "season", "teamID": "team",
                        "W": "wins", "HR": "home_runs"})
df["batting_avg"] = df["H"] / df["AB"]
df = df[["season", "team", "wins", "batting_avg", "home_runs"]].dropna()

print(f"Rows (team-seasons): {len(df)}")          # expect 150
print(df.describe().round(3), "\n")

# ---------------------------------------------------------------- 3. Per season
# This replicates your slides: one correlation per season for each variable.
rows = []
for season, g in df.groupby("season"):
    r_avg, p_avg = stats.pearsonr(g["batting_avg"], g["wins"])
    r_hr, p_hr = stats.pearsonr(g["home_runs"], g["wins"])
    rows.append({"season": season,
                 "r_batting_avg": r_avg, "p_batting_avg": p_avg,
                 "r_home_runs": r_hr, "p_home_runs": p_hr})
by_season = pd.DataFrame(rows).round(3)
print(by_season, "\n")
print("Average r, batting avg:", by_season["r_batting_avg"].mean().round(3))
print("Average r, home runs:  ", by_season["r_home_runs"].mean().round(3), "\n")

# ---------------------------------------------------------------- 4. Pooled
r_avg, p_avg = stats.pearsonr(df["batting_avg"], df["wins"])
r_hr, p_hr = stats.pearsonr(df["home_runs"], df["wins"])
print(f"Pooled batting avg vs wins: r = {r_avg:.3f}, p = {p_avg:.4f}")
print(f"Pooled home runs vs wins:   r = {r_hr:.3f}, p = {p_hr:.4f}\n")

# ---------------------------------------------------------------- 5. Regression
# Both predictors in one model, standardized so the coefficients are comparable.
# C(season) adds a fixed effect so each season's league-wide level is controlled.
for col in ["wins", "batting_avg", "home_runs"]:
    df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()

model = smf.ols("wins_z ~ batting_avg_z + home_runs_z + C(season)", data=df).fit()
print(model.summary())

# Do the two predictors overlap? (If highly correlated, separate analyses mislead.)
r_between, _ = stats.pearsonr(df["batting_avg"], df["home_runs"])
print(f"\nCorrelation between batting avg and home runs: {r_between:.3f}")

# ---------------------------------------------------------------- 6. Plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, col, label in [(axes[0], "batting_avg", "Team batting average"),
                       (axes[1], "home_runs", "Team home runs")]:
    ax.scatter(df[col], df["wins"], alpha=0.6)
    slope, intercept, *_ = stats.linregress(df[col], df["wins"])
    xs = df[col].sort_values()
    ax.plot(xs, intercept + slope * xs, color="red")
    ax.set_xlabel(label)
    ax.set_ylabel("Wins")
    ax.set_title(f"{label} vs. wins (2018-2023, excl. 2020)")
plt.tight_layout()
plt.savefig("mlb_wins_scatter.png", dpi=200)
plt.show()