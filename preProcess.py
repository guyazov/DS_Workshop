import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt

# Change positions to float numbers
def preProcess_position(database):
    database['Pos'] = database['Pos'].replace(['PG', 'SG'], 0.8)
    database['Pos'] = database['Pos'].replace(['PF', 'SF'], 0.5)
    database['Pos'] = database['Pos'].replace(['C'], 0.2)
    return database
