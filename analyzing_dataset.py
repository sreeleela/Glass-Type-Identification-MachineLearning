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

warnings.filterwarnings("ignore")								#remove this

# Load dataset
url = "glass.csv"
names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'class']
dataset = pandas.read_csv(url, names=names)
# shape
print("Dataset Shape: ",dataset.shape)
# head
print("First 20 Rows....")
print(dataset.head(20))
# descriptions
print("Description of Data...")
print(dataset.describe())
# class distribution
print("Class Distribution")
print(dataset.groupby('class').size())

#Analyzing Data
#Box Plot
f, (m1, m2, m3) = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(8,5))
m1.boxplot(dataset.RI);
m1.set_xlabel("RI");
m2.boxplot(dataset.Na);
m2.set_xlabel("Na");
m3.boxplot(dataset.Mg);
m3.set_xlabel("Mg");
plt.savefig("images/Boxplot1.png")
f, (m4, m5, m6) = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(8,5))
m4.boxplot(dataset.Al);
m4.set_xlabel("Al");
m5.boxplot(dataset.Si);
m5.set_xlabel("Si");
m6.boxplot(dataset.K);
m6.set_xlabel("K");
plt.savefig("images/Boxplot2.png")
f, (m7, m8, m9) = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(8,5))
m7.boxplot(dataset.Ca);
m7.set_xlabel("Ca");
m8.boxplot(dataset.Ba);
m8.set_xlabel("Ba");
m9.boxplot(dataset.Fe);
m9.set_xlabel("Fe");
plt.savefig("images/Boxplot3.png")
#Histograms
f, (m1, m2, m3) = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(8,5))
m1.hist(dataset.RI);
m1.set_xlabel("RI");
m2.hist(dataset.Na);
m2.set_xlabel("Na");
m3.hist(dataset.Mg);
m3.set_xlabel("Mg");
plt.savefig("images/Histogram1.png")
f, (m4, m5, m6) = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(8,5))
m4.hist(dataset.Al);
m4.set_xlabel("Al");
m5.hist(dataset.Si);
m5.set_xlabel("Si");
m6.hist(dataset.K);
m6.set_xlabel("K");
plt.savefig("images/Histogram2.png")
f, (m7, m8, m9) = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(8,5))
m7.hist(dataset.Ca);
m7.set_xlabel("Ca");
m8.hist(dataset.Ba);
m8.set_xlabel("Ba");
m9.hist(dataset.Fe);
m9.set_xlabel("Fe");
plt.savefig("images/Histogram3.png")
#scater plot
df = pd.DataFrame(dataset, columns=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
scatter_matrix(df)
plt.savefig("images/ScatterPlot.png")
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.savefig("images/CorrelationMatrix.png")