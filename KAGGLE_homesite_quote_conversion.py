'''
Created on 22.11.2015

@author: Tobias
'''
import pandas as pd
import numpy as np
from time import time
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier
from mlxtend.classifier import EnsembleClassifier
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

seed = 260681
print("Reading files")
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

print("Preprocessing")
y = train.QuoteConversion_Flag.values
train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
test = test.drop('QuoteNumber', axis=1)

# Lets play with some dates
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek


test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek

train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)

train = train.fillna(-1)
test = test.fillna(-1)
header = train.columns.values.tolist()

for f in train.columns:
    if train[f].dtype=='object':
        #print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

#feature importance
# print("Feature selection")
# forest = ExtraTreesClassifier(n_estimators=100, random_state=0,bootstrap=True)
# forest.fit(train,y)
# n_feat = 50
# importances = forest.feature_importances_
# indices = np.argsort(importances)[::-1]
# train = train.iloc[:,indices[0:n_feat]]
# test = test.iloc[:,indices[0:n_feat]]
# for f in range(n_feat):
#     print("%d. feature %s (%f)" % (f + 1, header[indices[f]], importances[indices[f]]))                

#PCA
pca = PCA(n_components=30).fit(train)
reduced_train = pca.transform(train)
test = pca.transform(test)

print("Training")
# clf = GradientBoostingClassifier(n_estimators=200,learning_rate=0.025,max_depth=10,subsample=0.8)
# gb_model = clf.fit(train, y) 
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1.0 ,5e2, 1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(reduced_train, y)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


print("Predicting")
preds = clf.predict_proba(test)[:,1]
sample = pd.read_csv('input/sample_submission.csv')
sample.QuoteConversion_Flag = preds
sample.to_csv('submission.csv', index=False)
