import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
import numpy as np

def analyze_shots_per_game(database):
    '''
    Description: Playoffs vs regular season plotting.
    :param database: a pandas Dataframe
    :return: Nothing.
    '''
    shot_attempted_per_game = database.groupby(["season", "playoffs"])['shot_made'].count().unstack()
    shot_made_per_game = database.groupby(["season", "playoffs"])['shot_made'].sum().unstack()

    # this has to be divided by the number of games for each season to get an average
    number_of_games = database.groupby(["season", "playoffs"])['game_id'].nunique().unstack()

    average_shot_made_per_game = shot_made_per_game / number_of_games
    average_shot_attempted_per_game = shot_attempted_per_game / number_of_games

    f, (ax1) = plt.subplots(figsize=(18, 18))
    first = average_shot_attempted_per_game.plot(ax=ax1, marker='o', figsize=(15, 8), xticks=range(10),
                                                 color=['b', 'r'], rot=90)
    second = average_shot_made_per_game.plot(ax=ax1, marker='o', linestyle='--', figsize=(15, 8), xticks=range(10),
                                             color=['b', 'r'], rot=90)
    ax1.set_title('Average number of free throws per period. Attempted vs Successful', size=25)
    legend = plt.legend((' playoffs attempted', 'regular attempted', 'playoffs successful', 'regular successful'),
                        loc=6)
    ax1.add_artist(legend)
    plt.savefig("img/shots_per_game.png")
    plt.show()

def find_and_plot_correlations(database):
    '''
    Description: plot correlation heat map.
    :param database: a pandas Dataframe.
    :return: Nothing.
    '''
    plt.figure(figsize=(22, 22))
    cor = database.corr()
    
    ax = sns.heatmap(cor, annot=True,fmt='.2f', cmap=plt.cm.Reds,annot_kws={"fontsize":28})
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.tight_layout()
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 30)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 30)
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90) 
    plt.show()


def plot_confusion(classifier, X_test, y_test, class_names):
    '''
    Description: plot the confusion matrix
    :param classifier: the classifier
    :param X_test: x test dataset
    :param y_test: y(labels) test dataset
    :param class_names: the names of the classes
    :return: Nothing.
    '''
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(estimator=classifier, X=X_test,
                                     y_true=y_test,
                                     display_labels=class_names,
                                     normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()


def plot_score_percentage_per_position(database):
    # Throws percentage per position
    ax2 = sns.catplot(x="Pos", y="FT%", data=database)
    ax2.set(xlabel='Player Position', ylabel='Score Percentage')
    plt.show()


def plot_throws_statistics(database):
    '''
    Description: Plot throws statistics.
    :param database: a pandas Dataframe.
    :return: Nothing.
    '''
    minutes = range(int(max(database.abs_min)))
    total_throws = []
    success_throws = []
    success_precentage = []
    success_precentage_by_rank = []
    success_precentage_by_scoreDif = []
    difs = range(int(np.min(database['Difference'])) + 1, int(np.max(database['Difference'])) + 1)
    ranks = range(1, int(np.nanmax(database['draftRank'])) + 1)

    def count_throws(database2, minute2):
        made = len(database2[(database2.abs_min == minute2) &
                             (database2.shot_made == 1)])
        success_throws.append(made)
        total = len(database2[database2.abs_min == minute2])
        total_throws.append(total)
        if total == 0:
            precentage = 0.0
        else:
            precentage = made / total
        success_precentage.append(precentage)

    def throws_per_rank(database2, rank2):
        total = len(database2[database2.draftRank == rank2])
        if total == 0:
            precentage = 0.0
        else:
            made = len(database2[(database2.draftRank == rank2) & (database2.shot_made == 1)])
            precentage = made / total
        success_precentage_by_rank.append(precentage)
    for minute in minutes:
        count_throws(database, minute)
    for rank in ranks:
        throws_per_rank(database, rank)

    def throws_per_dif(database2, dif2):
        total = len(database2[database2.Difference == dif2])
        if total == 0:
            precentage = 0.0
        else:
            made = len(database2[(database2.Difference == dif2) & (database2.shot_made == 1)])
            precentage = made / total
        success_precentage_by_scoreDif.append(precentage)
    for dif in difs:
        throws_per_dif(database, dif)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(15, 15))
    ax1.plot(minutes, total_throws)
    ax1.title.set_text('Number of throws over time')
    ax1.set_xlim([1, 48])
    ax1.set_ylim([0, 40000])
    ax1.plot([12, 12], [0, 40000], '--', linewidth=1, color='r')
    ax1.plot([24, 24], [0, 40000], '--', linewidth=1, color='r')
    ax1.plot([36, 36], [0, 40000], '--', linewidth=1, color='r')
    ax1.plot([48, 48], [0, 40000], '--', linewidth=1, color='r')
    ax1.set_xlabel('Minute')
    ax1.set_ylabel('num of throws')
    ax2.plot(minutes, success_precentage)
    ax2.title.set_text('Scoring % over time - Always worse at the beginning')
    ax2.set_xlim([1, 48])
    ax2.set_ylim([0.65, 0.85])
    ax2.plot([12, 12], [0, 1], '--', linewidth=1, color='r')
    ax2.plot([24, 24], [0, 1], '--', linewidth=1, color='r')
    ax2.plot([36, 36], [0, 1], '--', linewidth=1, color='r')
    ax2.plot([48, 48], [0, 1], '--', linewidth=1, color='r')
    ax2.set_xlabel('Minute')
    ax2.set_ylabel('Free Throws %')
    ax3.plot(list(ranks), success_precentage_by_rank)
    ax3.title.set_text('Scoring % as function of Draft Rank')
    ax3.set_ylim([0, 1])
    ax3.set_xlabel('Draft Rank')
    ax3.set_ylabel('Free Throws %')
    ax4.bar(list(difs), success_precentage_by_scoreDif)
    ax4.title.set_text('Success Throws percentage as function of Score Difference')
    ax4.set_ylim([0, 1])
    ax4.set_xlabel('Score Difference')
    ax4.set_ylabel('Free Throws %')
    plt.tight_layout()

def plot_distributions(database):
    '''
    Description: Plot parameters distributions.
    :param database: a pandas Dataframe.
    :return: Nothing.
    '''
    sns.kdeplot(database['draftRank'], label="draftRank")
    sns.kdeplot(database['Height'], label="Height")
    sns.kdeplot(database['Weight'], label="Weight")
    sns.kdeplot(database['Difference'], label="Difference")
    plt.legend();
    plt.show()

def plot_distributions_of_shooting_percentage_only(database):
    '''
    Description: Plot shooting percentage parameters distributions.
    :param database: a pandas Dataframe.
    :return: Nothing.
    '''

    sns.kdeplot(database['FT%'], label="FT%")
    sns.kdeplot(database['2P%'], label="2P%")
    sns.kdeplot(database['3P%'], label="3P%")
    sns.kdeplot(database['FG%'], label="FG%")
    plt.legend();
    plt.show()


def plot_general_info(database):
    '''
    Description: plot general info about features such as shooting hand,
    and scoring % per position.
    :param database: a pandas Dataframe.
    :return: Nothing.
    '''
    plt.bar([1, 2], [database.groupby(["ShootingHand"])["player"].nunique()[0],
                   database.groupby(["ShootingHand"])["player"].nunique()[1]], width=0.1,
            tick_label=['Right', 'Left'], label='ShootingHand')
    plt.legend(loc=2)
    plt.show()

