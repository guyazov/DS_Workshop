from sklearn import linear_model
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def logreg(X_train, Y_train,X_test, Y_test):
    '''
    Description: logistic regression model
    :param X_train: ndarray of x_train
    :param Y_train: ndarray of y_train
    :param X_test: ndarray of x_test
    :param Y_test: ndarray of y_test
    :return: results of the logistic regression model.
    '''
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X_train, Y_train)
    predictions = logreg.predict(X_test)
    predicted_probs = logreg.predict_proba(X_test)
    return metrics.classification_report(Y_test, predictions)


def SVM(X_train, Y_train, X_test, Y_test, kernel):
    '''
    Description: SVM model
    :param X_train: ndarray of x_train
    :param Y_train: ndarray of y_train
    :param X_test: ndarray of x_test
    :param Y_test: ndarray of y_test
    :param kernel: the kernel of the SVM.
    :return: results of the SVM model.
    '''
    linear_svm_model = SVC(kernel=kernel, class_weight='balanced', verbose=True, C=1)
    linear_svm_model.fit(X_train.values, Y_train.values)
    predictions = linear_svm_model.predict(X_test.values)
    return metrics.classification_report(Y_test, predictions)

def svm_grid_search():
    '''
    Description: performing grid search on the SVM model parameters.
    :return: Nothing.
    '''
    Gammas = [1e-3, 1e-4]
    Cs = [1, 10, 100, 1000]
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': Gammas, 'C': Cs},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(), tuned_parameters)
    scores = [x['mean_test_score'] for x in clf.cv_results_]
    scores = np.array(scores).reshape(len(Cs), len(Gammas))
    for ind, i in enumerate(Cs):
        plt.plot(Gammas, scores[ind], label='C: ' + str(i))
    plt.legend()
    plt.xlabel('Gamma')
    plt.ylabel('Mean score')
    plt.show()


def nn_classifier(X_train, Y_train, X_test, Y_test):
    '''
    Description: neural network classifier model
    :param X_train: ndarray of x_train
    :param Y_train: ndarray of y_train
    :param X_test: ndarray of x_test
    :param Y_test: ndarray of y_test
    :return: results of the neural network classifier model.
    '''
    model = Sequential()
    # 14 is the number of features
    model.add(Dense(10, input_dim=14, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # in case of multi class classification one should use 'categorical_crossentropy'
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=100, batch_size=64)
    y_pred = model.predict_classes(X_test)
    score = model.evaluate(X_test, Y_test, verbose=1)
    acc = accuracy_score(y_pred, Y_test)
    print('Accuracy is:', acc * 100)
    print(score)


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


def logreg_grid_search(logreg_model,X_train, y_train, X_test, y_test, scoring):
    '''
    Description: performing grid search on the logistic regression model parameters.
    :param logreg_model: logistic regression model
    :param X_train: ndarray of x_train
    :param y_train: ndarray of y_train
    :param X_test: ndarray of x_test
    :param y_test: ndarray of y_test
    :param scoring: scoring
    :return:
    '''
    grid_values = {'penalty': ['l2'], 'C': [0.001, .009, 0.01, .09, 1, 5, 10, 25]}
    grid_clf_acc = GridSearchCV(logreg_model, param_grid=grid_values, scoring=scoring)
    grid_clf_acc.fit(X_train, y_train)

    # Predict values based on new parameters
    y_pred_acc = grid_clf_acc.predict(X_test)

    # New Model Evaluation metrics
    print('Accuracy Score : ' + str(accuracy_score(y_test, y_pred_acc)))
    print('Precision Score : ' + str(precision_score(y_test, y_pred_acc)))
    print('Recall Score : ' + str(recall_score(y_test, y_pred_acc, pos_label=0)))
    print('F1 Score : ' + str(f1_score(y_test, y_pred_acc)))



def random_forest(X_train, Y_train, X_test, Y_test):
    rf = RandomForestClassifier(n_estimators=16, max_depth=20, random_state=0)
    rf.fit(X_train.values, Y_train.values)
    predictions = rf.predict(X_test.values)
    return metrics.classification_report(Y_test, predictions)
