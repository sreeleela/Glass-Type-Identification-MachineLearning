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
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from plot_learning_curve import plot_learning_curve
from sklearn.svm import LinearSVC
from plot_cm import plot_confusion_matrix
import warnings

warnings.filterwarnings("ignore")								#remove this

# Load dataset
url = "glass.csv"
names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'class']
dataset = pandas.read_csv(url, names=names)

a_score = []
def crossValidation():
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,7))
	X = dataset[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
	X = np.array(X)
	X = min_max_scaler.fit_transform(X)
	Y = dataset["class"]
	Y = np.array(Y)
	
	nfold = 25
	precision = []
	recall = []
	fscore = []
	clf = linear_model.LogisticRegression()
	skf = model_selection.StratifiedKFold(n_splits=nfold)
	y_test_total = []
	y_pred_total = []
    
	for train_index, test_index in skf.split(X, Y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		y_test_total.extend(y_test.tolist())
		model = clf.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		y_pred_total.extend(y_pred.tolist())
		p,r,f,s = precision_recall_fscore_support(y_test, y_pred, average='weighted')
		#print(accuracy_score(y_test, y_pred))
		a_score.append(accuracy_score(y_test, y_pred))
		precision.append(p)
		recall.append(r)
		fscore.append(f)
	plot_learning_curve(clf, "Learning Curves", X, Y, ylim=None, cv=skf, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))
	plt.savefig('images/LR-LearningCurve.png')
	return pd.Series(y_test_total), pd.Series(y_pred_total), np.mean(precision),np.mean(recall),np.mean(fscore), np.mean(a_score)
	
classes = ['building float','building non float','vehicle float','containers','tableware','headlamps']
iterations = [i for i in range(0, 25)]
y_test, y_pred, precision, recall, fscore, accuracy = crossValidation()
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes, title='Confusion Matrix')
print("\nHere is the Classification Report:")
print(classification_report(y_test, y_pred, target_names=classes))
print("prfa:",precision, recall, fscore, accuracy)
plt.tight_layout()
plt.savefig("images/LR-ConfusionMatrix.png")
plt.figure()
plt.plot(iterations, a_score)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.savefig("images/LR-IterationsVsAccuracy.png")