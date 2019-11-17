import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt
df = pd.read_csv('SUMMARY.csv')

# Start of getting some intuition of the data
print(df.columns)
# Show right handed vs left handed
print(df["player"].nunique())
# Show right handed vs left handed
print(df.groupby(["ShootingHand"])["player"].nunique())
# End of getting some intuition of the data

# Start of missing data statistics
f, (ax1) = plt.subplots(figsize=(18, 18))
missing_data = df.isnull().sum() * 100 / len(df)
print(missing_data)
missing_data.plot(marker='o', figsize=(18, 7), xticks=range(21))
ax1.set_title('Missing Data Percentage By Category)', size=25)
# End of missing data statistics

shot_attempted_per_game = df.groupby(["season", "playoffs"])['shot_made'].count().unstack()
shot_made_per_game = df.groupby(["season", "playoffs"])['shot_made'].sum().unstack()

# this has to be divided by the number of games for each season to get an average
number_of_games=df.groupby(["season", "playoffs"])['game_id'].nunique().unstack()

average_shot_made_per_game = shot_made_per_game/number_of_games
average_shot_attempted_per_game = shot_attempted_per_game/number_of_games


f, (ax1) = plt.subplots(figsize=(18,18))
first=average_shot_attempted_per_game.plot(ax=ax1, marker='o', figsize=(15,8), xticks=range(10), color=['b','r'], rot=90)
second=average_shot_made_per_game.plot(ax=ax1, marker='o', linestyle='--', figsize=(15,8), xticks=range(10), color=['b','r'], rot=90)
ax1.set_title('Average number of free throws per period. Attempted vs Successful)', size=25)
legend=plt.legend((' playoffs attempted','regular attempted','playoffs successful','regular successful'), loc=6, prop={'size': 15})
ax1.add_artist(legend)
plt.show()
