import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt
import xlrd
import os

# Change positions to float numbers
def preProcess_position(database):
    database['Pos'] = database['Pos'].replace(['PG', 'SG'], 0.8)
    database['Pos'] = database['Pos'].replace(['PF', 'SF'], 0.5)
    database['Pos'] = database['Pos'].replace(['C'], 0.2)
    return database

def Dataset_FT_Percentage_per_player(database):
    players = database["player"].unique()
    FT_dict = dict()
    for player in players:
        shots_made = len(database[(database.player == player) & (database.shot_made == 1)])
        shots_total = len(database[(database.player == player)])
        FT_dict[player] = shots_made/shots_total
    return FT_dict


def Overall_FT_Percentage_per_player(database):
    players = database["player"].unique()
    FT_dict_2 = dict()
    df = database.groupby("player")["FT%"].unique()
    FT_dict_2 = df.to_dict()
    return FT_dict_2

def calculateMSE(database):
    players = database["player"].unique()
    dataset_percentage_dict = Dataset_FT_Percentage_per_player(players, database)
    overall_percentage_dict = Overall_FT_Percentage_per_player(players, database)
    mse = 0
    for player in players:
        mse += ((overall_percentage_dict[player][0] - dataset_percentage_dict[player]) ** 2)
    mse = mse / len(players)
    return mse


def get_players_years_dict(database):
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
            if result.size==0:
                print("Failed player: ", player, year)
                failed_players.append(player)
                break
            single_player_dict[year.replace(" ","")] = result.values
        players_dict[player] = single_player_dict
    print("Failed players : ", failed_players)
    print("Players with no excel file : ", players_with_no_excel)
    return players_dict


def Align_teams_names_aux(teams_in_year,origs,replacements):
    teams_in_year = teams_in_year.tolist()
    for i, orig in enumerate(origs,0):
        if orig in teams_in_year:
            teams_in_year[teams_in_year.index(orig)] = replacements[i]
    return teams_in_year


def Align_teams_names(teams_in_year):
    origs = ['PHO','NOK','WAS','GSW','NJN','UTA','NYK','SAS','NOH','NOP','BRK','CHO']
    replacements = ['PHX','NO','WSH','GS', 'NJ','UTAH','NY', 'SA', 'NO', 'NO','BKN', 'CHA']
    return Align_teams_names_aux(teams_in_year,origs,replacements)


def add_team_column(database):
    # Here we add a new column to the database - the team in which the player play in the shooting time.
    b = get_players_teams_by_years(database)

    def team(row):
        try:
            teams_in_game = row.game.split(' - ')
            teams_in_year = b[row['player']][row['season'].split(' - ')[0]]
            teams_in_year = Align_teams_names(teams_in_year)
            a = [i for i in teams_in_year if i in teams_in_game]
            # if len(a) > 1:
            #    print("Very bad",row['player'], row['season'], a)
            return a[0]
        except:
            if row.game != 'EAST - WEST' and row.game != 'WEST - EAST':
                # print(row['player'],row['season'],teams_in_game)
                return np.nan
            else:
                return 'Allstar'

    database['Team'] = database.apply(team, axis=1)


def add_difference_column(database):
    def get_difference_by_team(row):
        try:
            if row['Team'] != 'Allstar' and row['Team'] != np.nan:
                team_index = row.game.split(' - ').index(row['Team'])
                return int(row.score.split(' - ')[team_index]) - int(row.score.split(' - ')[1 - team_index])
        except:
            return row['player'], row['game'], row['Team'], row['season']

    database['Difference'] = database.apply(get_difference_by_team, axis=1)


def add_team_and_difference_columns(database):
    add_team_column(database)
    add_difference_column(database)
