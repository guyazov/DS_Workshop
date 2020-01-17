from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

v = model_database.player.value_counts()
rf_database = model_database[model_database.player.isin(v.index[v.gt(200)])] ## leave only players with more than 200 games.
rf = RandomForestClassifier(n_estimators=10, max_depth=8, random_state=0)
Y = rf_database['shot_made']
X = rf_database.drop(columns=['shot_made','player'])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
rf.fit(X_train.values, y_train.values)
predictions = rf.predict(X_train.values)
print(metrics.classification_report(y_train, predictions))

#Trying another threshold
threshold = 0.7
predicted_proba = rf.predict_proba(X_train.values)
predicted = (predicted_proba [:,1] >= threshold).astype('int')
print(metrics.classification_report(y_train, predicted))
