from sklearn import linear_model
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

def logreg(X_train, Y_train,X_test, Y_test):
    '''
    :param X_train: ndarray of x_train
    :param Y_train: ndarray of y_train
    :param X_test: ndarray of x_test
    :param Y_test: ndarray of y_test
    :return:
    '''
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X_train, Y_train)
    predictions = logreg.predict(X_test)
    predicted_probs = logreg.predict_proba(X_test)
    return metrics.classification_report(Y_test, predictions)


def SVM(X_train, Y_train, X_test, Y_test, kernel):
    linear_svm_model = SVC(kernel=kernel, class_weight='balanced', verbose=True, C=1)
    linear_svm_model.fit(X_train.values, Y_train.values)
    predictions = linear_svm_model.predict(X_test.values)
    return metrics.classification_report(Y_test, predictions)

def svm_grid_search():
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


def nn_classifier(X_train, Y_train, X_test, Y_test, kernel):
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
