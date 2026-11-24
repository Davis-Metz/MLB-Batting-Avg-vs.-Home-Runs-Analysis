# MLB-Batting-Avg-vs.-Home-Runs-Analysis

# Do MLB teams win more with a higher batting average or more home runs?

Analysis of 150 MLB team-seasons (2018-2023, excluding the 2020 shortened season).

## Question
Which is more strongly associated with team wins: team batting average or total home runs?

## Method
- Pearson correlations of each variable with wins, by season and pooled
- Linear regression of wins on both variables, with a season control
  (standardized variables, so coefficients are comparable)

## Results
- Home runs correlated more strongly with wins in 3 of 5 seasons
  (average r = 0.64 vs. 0.54 for batting average)
- Pooled: home runs r = 0.58, batting average r = 0.51
- Regression (R² = 0.52): home runs had the larger standardized effect (0.58 vs. 0.36);
  both were statistically significant (p < 0.001)

## Limitations
- Correlation does not show causation
- Pitching, defense, and ballpark also affect wins and are not in the model
- Batting average and home runs overlap (r = 0.43)

## How to run
1. Download Teams.csv (comma-delimited version) from https://sabr.org/lahman-database/
2. Put it in the same folder as the script
3. ` pip install pandas scipy statsmodels matplotlib`
4. Run `python mlb_wins_analysis.py`

Data: Lahman Baseball Database (SABR).
