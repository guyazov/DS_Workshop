from sklearn import linear_model
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

def logreg(X_train, Y_train,X_test, Y_test):
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X_train.values, Y_train.values)
    predictions = logreg.predict(X_test.values)
    predicted_probs = logreg.predict_proba(X_test.values)
    return metrics.classification_report(Y_test, predictions)


def SVM(X_train, Y_train, X_test, Y_test, kernel):
    linear_svm_model = SVC(kernel=kernel, class_weight='balanced', verbose=True, C=1)
    linear_svm_model.fit(X_train.values, Y_train.values)
    predictions = linear_svm_model.predict(X_test.values)
    return metrics.classification_report(Y_test, predictions)

def svm_grid_searc():
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

