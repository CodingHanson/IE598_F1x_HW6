import pandas as pd
import numpy as np
df = pd.read_csv('F:/MSFE/machine_learning/HW6/ccdefault.csv')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
y = df['DEFAULT']
X = df.drop(['ID','DEFAULT'],axis=1)
print('X:',X.head(),'y:',y.head())
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=16)
import time

#Classification Tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
start = time.clock()
SEED = range(1,11)
tree_scores = []
for i in SEED:
    dt = DecisionTreeClassifier(max_depth=6, random_state=i)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    tree_scores.append(metrics.accuracy_score(y_test,y_pred))

max_tree = tree_scores.index(max(tree_scores)) +1
max_tree_score = max(tree_scores)
mean_tree = np.mean(tree_scores)
std_tree = np.std(tree_scores)
end = time.clock()
print('It take', end-start,'s to run')
print("The best performance score for DecisionTree is {}".format(max_tree_score))
import matplotlib.pyplot as plt
plt.title('DecisionTree: Random State')
plt.plot(SEED, tree_scores, label = 'Accuracy Scores')
plt.legend()
plt.xlabel('SEED')
plt.ylabel('Default or not')
plt.show()
print(list(tree_scores))
print('The mean of the scores is', mean_tree, 'and its std is ',std_tree)

# K=10 for using cross_val_scores
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
start = time.clock()
dt = DecisionTreeClassifier(max_depth=6,random_state=16)
scores = cross_val_score(estimator=dt,X=X_train,y=y_train,cv=10,n_jobs=4)
end = time.clock()
print('It take', end-start,'s to run')
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))











print("-------------------------------------------------------------------------")
print("My name is Han Wang")
print("My NetID is: 'hanw8'")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")