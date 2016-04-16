from zipfile import ZipFile
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import preprocessing

z = ZipFile('data/train.csv.zip')
train = z.open('train.csv')
df = pd.read_csv(train)

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(df.Category.values)
X_train = df[['X', 'Y']].values
from sklearn import svm
clf = svm.LinearSVC()

print(datetime.now())
y_result = clf.fit_transform(X_train, y_train)
print(datetime.now())
s = clf.score(X_train, y_result)
print(datetime.now())

print(s)
