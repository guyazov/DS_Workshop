import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt
df = pd.read_csv('SUMMARY.csv')

# Start of getting some intuition of the data
print(df.columns)
# Start of fixing players missing stats
df.loc[df['player'] == "Nene", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]] =\
    ["Right", "C", "7","211","113","0.548","0.132","0.66","0.551"]
df.loc[df['player'] == "Antoine Walker", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "PF", "6","203","101","0.414","0.325","0.66","0.401"]
df.loc[df['player'] == "James Jones", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SF", "49","203","98","0.401","0.401","0.84","0.426"]
df.loc[df['player'] == "Mouhamed Sene", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "C", "10","211","104","0.427","","0.589","0.427"]
df.loc[df['player'] == "Anthony Johnson", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "PG", "39","190","86","0.414","0.356","0.745","0.431"]
df.loc[df['player'] == "Sasha Pavlovic", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SG", "19","203","99","0.404","0.346","0.673","0.433"]
df.loc[df['player'] == "Dajuan Wagner", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SG", "6","188","90","0.366","0.321","0.770","0.385"]
df.loc[df['player'] == "Marcus Vinicius", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SF", "43","203","102","0.457","0.421","0.556","0.500"]
df.loc[df['player'] == "Jose Juan", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "PG", "undrafted","178","83","0.424","0.351","0.792","0.463"]
df.loc[df['player'] == "Slava Medvedenko", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "PF", "","208","113","0.450","0.154","0.740","0.453"]
df.loc[df['player'] == "Luke Jackson", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Left", "SF", "10","208","113","0.357","0.360","0.732","0.356"]
df.loc[df['player'] == "Jeff Green", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SF", "5","203","106","0.440","0.334","0.850","0.482"]
df.loc[df['player'] == "Juan Carlos", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SG", "40","190","77","0.402","0.361","0.849","0.455"]
df.loc[df['player'] == "Luc Mbah", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "PF", "37","203","104","0.454","0.335","0.660","0.478"]
df.loc[df['player'] == "Bobby Brown", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "PG", "undrafted","188","79","0.379","0.317","0.806","0.425"]
df.loc[df['player'] == "Luc Richard", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%","player"]]\
    = ["Right", "PF", "37","203","104","0.454","0.335","0.660","0.478","Luc Mbah"]
df.loc[df['player'] == "Armon Johnson", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Left", "PG", "34","190","88","0.458","0.400","0.679","0.465"]
df.loc[df['player'] == "Pooh Jeter", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "PG", "undrafted","180","79","0.409","0.200","0.902","0.437"]
df.loc[df['player'] == "Tristan Thompson", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "PF", "4","206","107","0.520","0.200","0.610","0.521"]
df.loc[df['player'] == "Chris Johnson", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "C", "undrafted","211","95","0.562","","0.699","0.562"]
df.loc[df['player'] == "Markieff Morris", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "PF", "13","203","111","0.449","0.340","0.777","0.481"]
df.loc[df['player'] == "Metta World", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SF", "16","198","117","0.414","0.339","0.715","0.447"]
df.loc[df['player'] == "B.J. Mullens", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "C", "24","213","124","0.408","0.319","0.706","0.438"]
df.loc[df['player'] == "Marshon Brooks", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SG", "25","196","90","0.447","0.345","0.751","0.478"]
df.loc[df['player'] == "Marcus Morris", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "PF", "14","203","106","0.430","0.364","0.751","0.473"]
df.loc[df['player'] == "Trey Thompkins", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "C", "37","208","111","0.393","0.308","0.714","0.417"]
df.loc[df['player'] == "Harrison Barnes", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SF", "7","203","102","0.447","0.373","0.794","0.476"]
df.loc[df['player'] == "Greg Smith", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "C", "","208","113","0.617","0.0","0.576","0.618"]
df.loc[df['player'] == "J.J. Barea", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "PG", "undrafted","178","83","0.424","0.351","0.792","0.463"]
df.loc[df['player'] == "Michael Kidd-Gilchrist", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SF", "2","198","105","0.477","0.286","0.714","0.482"]
df.loc[df['player'] == "Nando de", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SF", "53","196","90","0.429","0.363","0.835","0.463"]
df.loc[df['player'] == "Moe Harkless", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SF", "15","201","99","0.477","0.321","0.614","0.545"]
df.loc[df['player'] == "Diante Garrett", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "PG", "","193","86","0.373","0.351","0.682","0.384"]
df.loc[df['player'] == "Jeff Ayres", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "C", "31","206","108","0.553","0.400","0.776","0.554"]
df.loc[df['player'] == "Slava Kravtsov", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "C", "","213","117","0.672","","0.333","0.672"]
df.loc[df['player'] == "Clint Capela", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "C", "25","208","108","0.636","0.0","0.522","0.636"]
df.loc[df['player'] == "P.J. Hairston", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SF", "26","198","104","0.343","0.295","0.810","0.414"]
df.loc[df['player'] == "James Michael", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "PF", "","206","104","0.528","0.294","0.534","0.545"]
df.loc[df['player'] == "Henry Walker", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SF", "47","198","99","0.446","0.369","0.760","0.560"]
df.loc[df['player'] == "Jonathon Simmons", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SG", "","198","88","0.443","0.317","0.756","0.487"]
df.loc[df['player'] == "Willie Reed", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "C", "","208","99","0.592","0.333","0.561","0.595"]
df.loc[df['player'] == "Xavier Munford", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "SG", "","188","81","0.398","0.360","0.500","0.414"]
df.loc[df['player'] == "W Russell", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
    = ["Right", "PG", "","183","77","0.347","0.308","0.636","0.354"]

df.to_csv("Test.csv")
# End of fixing players missing stats
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
