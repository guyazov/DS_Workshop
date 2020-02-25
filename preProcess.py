import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def make_binary_variables(database):
    '''
    Description: convert the playoffs and ShootingHand variables to binary ones.
    :param database: a pandas Dataframe
    :return: The edited Dataframe.
    '''
    database['playoffs'] = database['playoffs'].map(lambda x : 0 if x == 'regular' else 1)
    database['ShootingHand'] = database['ShootingHand'].map(lambda x : 0 if x == 'Right' else 1)
    return database


def make_numeric_variables(database):
    '''
    Description: convert the Height, Weight and draft rank variables to
    numeric ones.
    :param database: a pandas Dataframe
    :return: The edited Dataframe.
    '''
    database['Height'] = database['Height'].map(lambda x: int(x))
    database['Weight'] = database['Weight'].map(lambda x: int(x))
    database['draftRank'] = database.draftRank.apply(lambda x: int(float(x)))
    return database


def fill_missing_values(database):
    '''
    Description: Fill missing values of FT% and 3P% with their median. For missing
    draft ranks or for undrafted players we use the value '61' since there are only
    60 players which been drafted every year.
    :param database: a pandas Dataframe
    :return: The edited Dataframe.
    '''
    database['FT%'] = database['FT%'].fillna(database['FT%'].mode()[0])
    database['3P%'] = database['3P%'].fillna(database['FT%'].mode()[0])
    database['draftRank'] = database['draftRank'].replace(np.nan, 61)
    database['draftRank'] = database['draftRank'].replace("undrafted", 61)
    return database



def make_pos_column_as_one_hot(X_train,X_test):
    '''
    Description: make the pos column one-hot encoded
    :param X_train: a pandas Dataframe
    :param X_test: a pandas Dataframe
    :return: X_train and X_test with pos column one-hot encoded.
    '''
    onehot_pos_col = pd.get_dummies(X_train['Pos'],prefix='Pos')
    X_train = pd.concat([X_train,onehot_pos_col],axis=1)
    X_train = X_train.drop(columns=['Pos'])
    onehot_pos_col = pd.get_dummies(X_test['Pos'],prefix='Pos')
    X_test = pd.concat([X_test,onehot_pos_col],axis=1)
    X_test = X_test.drop(columns=['Pos'])
    return X_train,X_test


def data_normalization(semiNormal_parameters, nonNormal_parameters,X_train,X_test):
    '''
    Description: normalize the given parameters
    :param semiNormal_parameters: semi normal params
    :param nonNormal_parameters: non normal params
    :param X_train: a pandas Dataframe
    :param X_test: a pandas Dataframe
    :return: normalized version of X_train and X_test
    '''
    sc = StandardScaler()
    mm = MinMaxScaler()
    X_train[semiNormal_parameters] = sc.fit_transform(X_train[semiNormal_parameters])
    X_train[nonNormal_parameters] = mm.fit_transform(X_train[nonNormal_parameters])
    X_test[semiNormal_parameters] = sc.fit_transform(X_test[semiNormal_parameters])
    X_test[nonNormal_parameters] = mm.fit_transform(X_test[nonNormal_parameters])
    return X_train, X_test


def preProcess_position(database):
    '''
    Description: changing positions to float numbers,using our knowledge of the
    position distribution.
    :param database: a pandas Dataframe.
    :return: a pandas Dataframe with positions as float numbers.
    '''
    database.loc[database.Pos == 'PG'] = 0.8
    database.loc[database.Pos == 'SG'] = 0.8
    database.loc[database.Pos == 'PF'] = 0.5
    database.loc[database.Pos == 'SF'] = 0.5
    database.loc[database.Pos == 'C' ] = 0.2
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

def add_time_columns(database):
    '''
    Description: add new columns related to time- second, minute, absolute minute
    and absolute time
    :param database: a pandas Dataframe
    :return: return a pandas Dataframe with the new columns
    '''
    database['minute'] = database.time.apply(lambda x: int(x[:len(x)-3]))
    database['sec'] = database.time.apply(lambda x: int(x[len(x)-2:]))
    database['abs_min'] = 12 - database['minute']+12*(database.period -1)
    database['abs_time'] = 60*(database.abs_min-1) + 60 - database['sec']

def add_score_columns(database):
    '''
    Description: add new columns related to scores- scores and score difference
    :param database: a pandas Dataframe
    :return: return a pandas Dataframe with the new columns
    '''
    database['scores'] = database.score.replace(' - ', '-').apply(lambda x: x.split('-'))
    database['scoreDif'] = database.scores.apply(lambda x: abs(int(x[1])-int(x[0])))
    return database



def manually_edit_players(df):
    '''
    Description: here we keep all the players that we edited manually.
    :param df: a pandas Dataframe
    :return: Nothing.
    '''

    '''
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
    df.loc[df['player'] == "Alvin Williams", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
        = ["Right", "PG", "47","196","83","0.421","0.313","0.760","0.443"]
    df.loc[df['player'] == "Alan Williams", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
        = ["Right", "PF", "","203","120","0.506","0.0","0.626","0.509"]
    df.loc[df['player'] == "Dominique Jones", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
        = ["Right", "SG", "25","193","97","0.366","0.095","0.729","0.393"]
    df.loc[df['player'] == "Jason Thompson", ["Pos","FT%","draftRank"]] = ["PF", "0.657","12"]
    ## Added by brian
    df.loc[df['player'] == "Mike Dunleavy", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
        = ["Right", "SF", "3","206","104","0.441","0.377","0.803","0.481"]
    df.loc[df['player'] == "Jeff Pendergraph", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
        = ["Right", "PF", "31","206","108","0.553","0.4","0.776","0.554"]
    df.loc[df['player'] == "Gerald Henderson", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
        = ["Right", "SG", "12","196","97","0.423","0.327","0.793","0.460"]
    df.loc[df['player'] == "Walker Russell", ["ShootingHand", "Pos", "draftRank","Height","Weight","FG%","3P%","FT%","2P%"]]\
        = ["Right", "PG", "undrafted","183","77","0.347","0.308","0.636","0.354"]
    '''
