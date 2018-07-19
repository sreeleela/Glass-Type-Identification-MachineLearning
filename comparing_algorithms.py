# Load libraries
import pandas
import numpy as np
from pandas.plotting import scatter_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings("ignore")

# Load dataset
url = "glass.csv"
names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'class']
dataset = pandas.read_csv(url, names=names)


min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,7))
X = dataset[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
X = np.array(X)
X = min_max_scaler.fit_transform(X)
Y = dataset["class"]
Y = np.array(Y)

X_train,X_validation,Y_train,Y_validation = model_selection.train_test_split(X,Y,test_size=0.33)
seed = 7
scoring = 'accuracy'

models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('LinearDiscriminant', LinearDiscriminantAnalysis()))
models.append(('KNeighbors', KNeighborsClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Random forest', RandomForestClassifier()))
models.append(('NN', MLPClassifier()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: Mean: %f Std Deviation: %f" % (name, cv_results.mean(), cv_results.std())
	print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.savefig("images/comparision.png")