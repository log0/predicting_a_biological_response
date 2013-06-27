"""
This uses RandomForest from Scikit-learn.


"""

import csv
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))


if __name__ == '__main__':
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'
    
    test_data = [ i for i in csv.reader(file(test_file, 'rb')) ]
    test_data = test_data[1:] # remove header

    data = [ i for i in csv.reader(file(train_file, 'rb')) ]    
    data = data[1:] # remove header
    random.shuffle(data)

    X = np.array([ [ float(j) for j in i[1:] ] for i in data ])
    Y = np.array([ float(i[0]) for i in data ])

    train_cutoff = len(data) * 3/4

    X_train = X[:train_cutoff]
    Y_train = Y[:train_cutoff]
    X_test = X[train_cutoff:]
    Y_test = Y[train_cutoff:]

    classifier = RandomForestClassifier(n_estimators=1000, n_jobs=3)
    classifier = classifier.fit(X_train, Y_train)

    Y_predict = classifier.predict_proba(X_test)[:,1]

    print 'Logloss = %s' % (logloss(Y_predict, Y_test))
    
    X_submission = np.array([ [ float(j) for j in i ] for i in test_data ])
    Y_submission = classifier.predict_proba(X_submission)[:,1]
    
    f = file('random_forest.out.csv', 'wb')
    f.write('\n'.join([str(i) for i in Y_submission]))
    f.close()


