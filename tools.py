from PandasBasketball import pandasbasketball as pb
from PandasBasketball.errors import StatusCode404
import requests
from bs4 import BeautifulSoup, NavigableString
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from PandasBasketball import pandasbasketball as pb
BASE_URL = "https://www.basketball-reference.com"

def get_player_stats(playerName):
    '''
    Description: this function receive an NBA player name (as a string) and extract with a web-wrapper (Beatiful-Soup),
                 we extract more stats about our players in dataset to collect more information.
                 Data is collected from "https://www.basketball-reference.com" website.
    :param playerName: string that should be a NBA player name
    :return: A pandas dataframe with player stats received from website.
    '''
    
    playerName = playerName.replace(".", "").replace("'", "")
    playerCode = pb.generate_code(playerName)
    url = BASE_URL + f"/players/{playerCode[0]}/{playerCode}.html"
    r = requests.get(url)

    # If the page is not found, raise the error
    # else, return the data frame
    if r.status_code == 404:
        raise StatusCode404
    else:
        soup = BeautifulSoup(r.text, "html.parser")
        comment_table = soup.find(text=lambda x: isinstance(x, NavigableString) and "totals" in x)
        soup = BeautifulSoup(comment_table, "html.parser")
        table = soup.find("table", id="totals")
        df = get_data_master2(table, "player")
        soup = BeautifulSoup(r.text,"html.parser")
        ## Find weight and height from webscrapper
        height_and_weight = str(soup.find("span", {"itemprop": "weight"}).next_sibling)
        height, weight = height_and_weight.split(",")
        height, weight = height.replace("(", "").replace("cm", ""), weight.replace(")", "").replace("kg", "")
        # Find shooting hand from webscrapper
        shootingHand = pd.NaT
        for strong in soup.find_all("strong"):
            if strong.string != None and "Shoots:" in strong.string:
                shootingHand = str(strong.nextSibling).replace("\n", "").replace(" ", "")
                break;
        ## Find draft rank from webscrapper
        draftRank = np.nan
        for strong in soup.find_all("strong"):
            if strong.string != None and "Draft:" in strong.string:
                draftString = list(filter(lambda x: "overall" in x, list(strong.next_siblings)))
                if len(draftString) == 0: break
                draftString = draftString[0]
                draftStringWords = draftString.split()
                indexOfDraftRank = draftStringWords.index("overall),") - 1
                draftRank = filter(lambda x: x.isdigit(), draftStringWords[indexOfDraftRank])
                draftRank = int("".join(list(draftRank)))
                break

        carrerDF = df
        carrerDF = carrerDF.assign(Height=height)
        carrerDF = carrerDF.assign(Weight=weight)
        carrerDF = carrerDF.assign(ShootingHand=shootingHand)
        carrerDF = carrerDF.assign(draftRank=draftRank)
        return carrerDF


def get_data_master2(table, tdata):
    '''
    Description: an internal function originally from "PandasBasketball" python library, (original name get_data_master), 
                    but library functions has some bugs so we fixed them.
    '''

    columns = []
    heading = table.find("thead")
    heading_row = heading.find("tr")
    for x in heading_row.find_all("th"):
        columns.append(x.string)

    body = table.find("tbody")
    rows = body.find_all("tr")
    body = table.find("tfoot")
    rows += body.find_all("tr")
    data = []
    for row in rows:
        temp = []
        th = row.find("th")
        td = row.find_all("td")
        if th:
            if th.text == "" or "season" in th.text: continue
            temp.append(th.text)
        else:
            continue

        if tdata == "gamelog" or tdata == "n_days":
            for v in td:
                if v.text == "Inactive" or v.text == "Did Not Play" or v.text == "Did Not Dress":
                    temp.extend([""] * (len(columns) - 8))  # 8 is the min. number of columns all tables have
                    break
                else:
                    temp.append(v.text)
            data.append(temp)
        elif tdata == "player" or tdata == "team":
            for v in td:
                temp.append(v.text)
            data.append(temp)

    # Removes the table's mid headers
    if tdata == "gamelog" or tdata == "n_days":
        for l in data:
            if len(l) != len(columns):
                data.remove(l)

    df = pd.DataFrame(data)
    df.columns = columns

    return df



## CODE TO CREATE EXCELS STATS
def create_players_excels_stats():
    '''
    Description: Function originally used to create players excel's stats files under 'player_stats' directory.
                 The function iterates over all the player in Kaggle's database and uses 'get_player_stats' func to extract 
                 the stats with the web-wrapper.
    '''
    database = pd.read_csv("free_throws.csv")
    playersNames = database.player.unique()
    failedPlayers = []
    #delete this line above!
    numberOfPlayers = len(playersNames)
    i=1
    for playerName in np.sort(playersNames):
        try:
            dataFrame = get_player_stats(playerName)
            dataFrame.to_excel("{}.xlsx".format(playerName))
#             print("player {} from total of {} - Succeed!".format(i, numberOfPlayers))
        except Exception as e:
#             print("{} RAISED AN ERROR DURING SCRIPT".format(playerName))
#             print("player {} from total of {} - ERROR!".format(i, numberOfPlayers))
            failedPlayers.append(playerName)
        i+=1

    with open('failed_players.txt', 'w') as f:
        for item in failedPlayers:
            f.write("%s\n" % item)
