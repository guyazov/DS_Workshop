from sklearn import linear_model
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

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
    
    
def neural_net_model(X_train,y_model_train):
    '''
    Description: neural network classifier model
    :param X_train: ndarray of x_train
    :param Y_train: ndarray of y_train
    :return: A NN keras model.
    '''
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import LeakyReLU
    from keras.utils import to_categorical
    from keras.optimizers import Adagrad
    model = Sequential()
    model.add(Dense(12, input_dim=X_train.shape[1]))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(20))#, activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(10))#, activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(5))#, activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer=Adagrad(learning_rate=0.001), metrics=['accuracy'])
    return model
    
    
    
    
def focal_loss_nn_model(X_train,y_model_train,focal_loss):
    '''
    Description: neural network classifier model
    :param X_train: ndarray of x_train
    :param Y_train: ndarray of y_train
    :param X_test: ndarray of x_test
    :param Y_test: ndarray of y_test
    :return: A neural network model with focal loss.
    '''                        
    import keras
    from keras.models import model_from_json
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import LeakyReLU
    from keras.utils import to_categorical
    from keras.optimizers import Adagrad

    focal_loss_model = Sequential()
    focal_loss_model.add(Dense(12, input_dim=X_train.shape[1]))
    focal_loss_model.add(LeakyReLU(alpha=0.1))
    focal_loss_model.add(Dense(20))#, activation='relu'))
    focal_loss_model.add(LeakyReLU(alpha=0.1))
    focal_loss_model.add(Dropout(0.5))
    focal_loss_model.add(Dense(10))#, activation='relu'))
    focal_loss_model.add(LeakyReLU(alpha=0.1))
    focal_loss_model.add(Dropout(0.5))
    focal_loss_model.add(Dense(5))#, activation='relu'))
    focal_loss_model.add(LeakyReLU(alpha=0.1))
    focal_loss_model.add(Dense(1, activation='sigmoid'))
    focal_loss_model.compile(loss=[focal_loss], optimizer=Adagrad(learning_rate=0.001), metrics=['accuracy'])
    return focal_loss_model


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



def random_forest(X_train, Y_train, X_test, Y_test,threshold_flag=True):
    '''
    Description: random forest model
    :param threshold_flag: if true - find highest recall model with said
    constrains if false - use the default model.
    :param X_train: ndarray of x_train
    :param y_train: ndarray of y_train
    :param X_test: ndarray of x_test
    :param y_test: ndarray of y_test
    :return: results of the random forest model.
    '''
    rf = RandomForestClassifier(n_estimators=16, max_depth=20, random_state=0)
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_test)
    predicted_proba = rf.predict_proba(X_test)
    if threshold_flag:
        threshold = find_best_thershold(X_test, Y_test,rf)
        predicted = (predicted_proba[:, 1] >= threshold).astype('int')
        return metrics.classification_report(Y_test, predicted) , rf
    else:
        return metrics.classification_report(Y_test, predictions) , rf

    
def knn_grid_search(X_train, y_model_train,X_test,y_test):
    '''
    Description: performing grid search on the SVM model parameters.
    :return: two lists, one for training reports and the second for test reports.
    every item in each list is a text with the performance of the model.
    K range: 1-11
    '''
    training_reports = []
    test_reports = []
    for i in range(1,12):
        knn = KNeighborsClassifier(n_neighbors=i, metric='minkowski',p=2)
        nn.fit(X_train, y_model_train)
        training_reports.append(metrics.classification_report(y_model_train, knn.predict(X_train)))
        test_reports.append(metrics.classification_report(y_test, knn.predict(X_test)))
    return training_reports, test_reports
                            
def find_best_thershold(X_test, Y_test,model_after_fit):
    '''
    Description: Find the threshold that gets highest recall score on missed shots
    while still getting at least 0.7 accuracy score.
    :param X_test: ndarray of x_test
    :param y_test: ndarray of y_test
    :param model_after_fit: model after fit action
    :return: The best threshold.
    '''

    recall_score_test_max = 0
    threshold_test_max = 0
    acc_score_test_max = 0                       
    thresholds = np.arange(0.6,1,0.01)
    for threshold in thresholds:
        predicted_test_proba = model_after_fit.predict_proba(X_test)
        predicted_test = (predicted_test_proba [:,1] >= threshold).astype('int')
        recall_performance = recall_score(Y_test, predicted_test, pos_label=0)
        acc_performance = accuracy_score(Y_test, predicted_test)
        if recall_score_test_max < recall_performance  and acc_performance > 0.7:
            recall_score_test_max = recall_performance
            threshold_test_max = threshold
            acc_score_test_max = acc_performance             
    print(f"Optimal threshold is: {threshold_test_max}")
#     print("Model performance with this thr")
#     print(f"Label 0 recall: {recall_score_test_max}")
#     print(f"Accuracy: {acc_score_test_max}")
    return threshold_test_max

def plot_as_func_threshold(X_test, Y_test,model_after_fit):
    '''
    Description: Find the threshold that gets highest recall score on missed shots
    while still getting at least 0.7 accuracy score.
    :param X_test: ndarray of x_test
    :param y_test: ndarray of y_test
    :param model_after_fit: model after fit action
    :return: The best threshold.
    '''
    recalls = []
    accuracy = []
    thresholds = np.arange(0.0,1,0.05)
    for threshold in thresholds:
        predicted_test_proba = model_after_fit.predict_proba(X_test)
        predicted_test = (np.asarray(predicted_test_proba)[:,0] >= threshold).astype('int')
        recalls.append(recall_score(Y_test, predicted_test, pos_label=0))
        accuracy.append(accuracy_score(Y_test, predicted_test))
    
    plt.plot(thresholds,recalls)
    plt.plot(thresholds,accuracy)
    plt.legend(['Label 0 recall', 'Total Accuracy'], loc='right')
    plt.xlabel("Threshold for classifing to label 1")
    plt.show()
    return recalls, accuracy

def evaluate_model(model,X_train,y_model_train,X_test,y_test,rf_model_flag=False):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    if(rf_model_flag):
        n_nodes = []
        max_depths = []

        # Stats about the trees in random forest
        for ind_tree in model.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)

        print(f'Average number of nodes {int(np.mean(n_nodes))}')
        print(f'Average maximum depth {int(np.mean(max_depths))}')

    # Training predictions (to demonstrate overfitting)
    train_predictions = model.predict(X_train.values)
    train_probs = model.predict_proba(X_train.values)[:, 1]

    # Testing predictions (to determine performance)
    predictions = model.predict(X_test.values)
    probs = model.predict_proba(X_test.values)[:, 1]



    # Plot formatting
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.size'] = 18
    
    baseline = {}
    
    baseline['recall'] = recall_score(y_test, 
                                     [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test, 
                                      [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(y_test, predictions)
    results['precision'] = precision_score(y_test, predictions)
    results['roc'] = roc_auc_score(y_test, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(y_model_train, train_predictions)
    train_results['precision'] = precision_score(y_model_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_model_train, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();