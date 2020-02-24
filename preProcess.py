import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
import os

def preProcess_position(database):
    '''
    Description: changing positions to float numbers
    :param database: a pandas Dataframe.
    :return: a pandas Dataframe with positions as float numbers.
    '''
    database.loc[database.Pos == 'PG'] = 0.8
    database.loc[database.Pos == 'SG'] = 0.8
    database.loc[database.Pos == 'PF'] = 0.5
    database.loc[database.Pos == 'SF'] = 0.5
    database.loc[database.Pos == 'C' ] = 0.2

#     database['Pos'] = database['Pos'].replace(['PG', 'SG'], 0.8)
#     database['Pos'] = database['Pos'].replace(['PF', 'SF'], 0.5)
#     database['Pos'] = database['Pos'].replace(['C'], 0.2)
    return database


def Dataset_FT_Percentage_per_player(database):
    '''
    Description: calculating free throw % of each player.
    :param database: a pandas Dataframe.
    :return: a dictionary with player as key and FT % as value.
    '''
    players = database["player"].unique()
    FT_dict = dict()
    for player in players:
        shots_made = len(database[(database.player == player) & (database.shot_made == 1)])
        shots_total = len(database[(database.player == player)])
        FT_dict[player] = shots_made/shots_total
    return FT_dict


def get_players_years_dict(database):
    '''
    Description: finding each player years of play.
    :param database: a pandas Dataframe.
    :return: a dictionary with player as key and its years of play in our dataset as value.
    '''
    players = database["player"].unique()
    players_years_of_play = {}
    for player in players:
        database[(database.player == player)].season.replace(' - ', '-').apply(lambda x: x.split('-'))
        seasons = []
        for season in database[(database.player == player)].season.unique():
            seasons.append(str(season).split('-')[0])
        players_years_of_play[player] = seasons
    return players_years_of_play


def get_players_teams_by_years(database):
    '''
    Description: finding the teams of each player (adjusted to its years of play).
    :param database: a pandas Dataframe.
    :return: a dictionary with player as key and value of another dictionary with year as key and
    the teams the player play in that year as value.
    '''
    players = database["player"].unique()
    players_years_of_play = get_players_years_dict(database)
    players_dict = {}
    failed_players = []
    players_with_no_excel = []
    FILE_ENDING_CONST = '.xlsx'
    for player in players:
        single_player_dict = dict()
        player_file_path = os.path.join("players_stats", str(player)+str(FILE_ENDING_CONST))
        if os.path.exists(player_file_path):
            dataFrame = pd.read_excel(player_file_path)
        else:
            players_with_no_excel.append(player)
            failed_players.append(player)
            continue
        years_of_play = players_years_of_play[player]
        for year in years_of_play:
            result = dataFrame[dataFrame.Season.str.contains(str(year.replace(" ","")),na=False)]['Tm']
            if result.size == 0:
                failed_players.append(player)
                break
            single_player_dict[year.replace(" ","")] = result.values
        players_dict[player] = single_player_dict
    return players_dict


def Align_teams_names_aux(teams_in_year,origs,replacements):
    '''
    Description: some teams held different names through the dataset. So we were needed to align them.
                 This method is an auxilary method which does that.
    :param teams_in_year: the teams we have at the moment
    :param origs: the teams names to be replaced.
    :param replacements: the teams names that shall replace other names.
    :return: a list of teams with coherent names.
    '''
    teams_in_year = teams_in_year.tolist()
    for i, orig in enumerate(origs,0):
        if orig in teams_in_year:
            teams_in_year[teams_in_year.index(orig)] = replacements[i]
    return teams_in_year


def Align_teams_names(teams_in_year):
    '''
    Description: call the auxilary function with the appropriate names.
    :param teams_in_year: a list of teams with maybe not coherent names.
    :return: a list of teams with coherent names.
    '''
    origs = ['PHO','NOK','WAS','GSW','NJN','UTA','NYK','SAS','NOH','NOP','BRK','CHO']
    replacements = ['PHX','NO','WSH','GS', 'NJ','UTAH','NY', 'SA', 'NO', 'NO','BKN', 'CHA']
    return Align_teams_names_aux(teams_in_year,origs,replacements)


def add_team_column(database):
    '''
    Description: adding a new column to the database - the team in which the player play in the shooting time.
    :param database: a pandas Dataframe.
    :return: Nothing.
    '''

    b = get_players_teams_by_years(database)

    def team(row):
        try:
            teams_in_game = row.game.split(' - ')
            teams_in_year = b[row['player']][row['season'].split(' - ')[0]]
            teams_in_year = Align_teams_names(teams_in_year)
            a = [i for i in teams_in_year if i in teams_in_game]
            return a[0]
        except:
            if row.game != 'EAST - WEST' and row.game != 'WEST - EAST':
                # print(row['player'],row['season'],teams_in_game)
                return np.nan
            else:
                return 'Allstar'

    database['Team'] = database.apply(team, axis=1)


def add_difference_column(database):
    '''
    Description: adding a new column to the database - the difference in the score (not absolute) in the
    shooting time.
    :param database: a pandas Dataframe.
    :return: Nothing.
    '''
    def get_difference_by_team(row):
        try:
            if row['Team'] != 'Allstar' and row['Team'] != np.nan and row['Team'] != 'nan':
                team_index = row.game.split(' - ').index(row['Team'])
                return int(row.score.split('-')[team_index].replace(' ', ''))\
                       - int(row.score.split('-')[1 - team_index].replace(' ', ''))
        except:
            print("game: ", row.game.split(' - '))
            print("score :", row['score'])
            print("score type :", type(row['score']))
            print("splitted: ", row.score.split('-'))
            print(row['player'], row['game'], row['Team'], row['season'], row['score'])
            return np.nan

    database['Difference'] = database.apply(get_difference_by_team, axis=1)


def add_team_and_difference_columns(database):
    '''
    Description: call the methods that add new team and difference columns.
    :param database: a pandas Dataframe.
    :return: Nothing.
    '''
    add_team_column(database)
    add_difference_column(database)

def creating_the_complete_db(database):
    '''
    Description: create an almost final version of our database.
    :param database: a pandas Dataframe.
    :return: Nothing.
    '''
    database['score'] = database['score'].astype(str)
    database['score'] = database['score'].replace(' - ', '-')
    database.drop(database.loc[(database['player'] == 'Scott Machado')].index, inplace=True)
    add_team_and_difference_columns(database)
    # Drop Allstar games
    database = database[database.Team != 'Allstar']

    database_p1 = database[:309009]
    print(database_p1.shape)
    database_p2 = database[309009:]
    print(database_p2.shape)
    database_p1.to_csv("complete_database.csv")
    database_p2.to_csv("complete_database_part2.csv")


# PCA for n dimensions; Should be used after normalizing the data
def run_PCA(X_train, X_test, n):
    '''
    Description: run PCA on the data
    :param X_train: train dataset.
    :param X_test: test dataset.
    :param n: number of dimensions we want reduce to.
    :return: train dataset and test dataset in lower dimension(n-th dimension).
    '''
    pca = PCA(n_components=n)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test


def find_correlations(database):
    '''
    Description: plot correlation heat map.
    :param database: a pandas Dataframe.
    :return: Nothing.
    '''
    plt.figure(figsize=(12, 10))
    cor = database.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()


def detect_and_remove_outliers(database, Y):
    '''
    Description: detect and remove outliers.
    :param database: a pandas Dataframe.
    :param Y: labels.
    :return: database and labels which outliers were removed from.
    '''
    indexes = (np.abs(stats.zscore(database.select_dtypes(exclude='object'))) < 3).all(axis=1)
    database = database[indexes]
    Y = Y[indexes]
    return database, Y

def add_previous_shots_feature(database_in):
    '''
    Description: adding new features - what shot is it and were previous
    shots were in or out.
    :param database_in: a pandas Dataframe.
    :return: a pandas Dataframe with the new features.
    '''
    database2 = database_in.copy()
    print(database2.shape)
    database2['First_shot'] = 0
    database2['Second_shot'] = 0
    database2['Third_shot'] = 0
    database2['First_shot_was_in'] = 0
    database2['Second_shot_was_in'] = 0

    first_player = database2.head(1)['player'].values[0]
    database2.loc[0, 'First_shot'] = 1
    if 'makes free throw 1' in database2.head(1)['play'].values[0]:
        score_first_flag = 1
    else:
        score_first_flag = -1
    if 'of 1' in database2.head(1)['play'].values[0]:
        number_of_remain_throws = 1
    if 'of 2' in database2.head(1)['play'].values[0]:
        number_of_remain_throws = 2
    if 'of 3' in database2.head(1)['play'].values[0]:
        number_of_remain_throws = 3

    for i, row in database2.iterrows():
        if i % 10000 == 0:
            print(i)
        # Are we still checking the same player?
        if row['player'] == first_player and number_of_remain_throws > 0:
            # 2nd shot
            if 'makes free throw 2' in row['play']:
                database2.loc[i, 'First_shot_was_in'] = score_first_flag
                database2.loc[i, 'Second_shot'] = 1
                score_second_flag = 1
            if 'misses free throw 2' in row['play']:
                database2.loc[i, 'First_shot_was_in'] = score_first_flag
                database2.loc[i, 'Second_shot'] = 1
                score_second_flag = -1
            # 3rd shot
            if 'free throw 3' in row['play']:
                database2.loc[i, 'First_shot_was_in'] = score_first_flag
                database2.loc[i, 'Second_shot_was_in'] = score_second_flag
                database2.loc[i, 'Third_shot'] = 1

        # New Player, or Same player - different shot cluster!
        else:
            first_player = row['player']
            database2.loc[i, 'First_shot'] = 1
            if 'makes free throw 1' in row['play']:
                score_first_flag = 1
            else:
                score_first_flag = -1
            if 'of 1' in row['play']:
                number_of_remain_throws = 1
            if 'of 2' in row['play']:
                number_of_remain_throws = 2
            if 'of 3' in row['play']:
                number_of_remain_throws = 3
        number_of_remain_throws -= 1
    return database2
