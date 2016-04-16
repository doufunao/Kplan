from zipfile import ZipFile
import pandas as pd
import numpy as np
from datetime import datetime

z = ZipFile('data/train.csv.zip')
train = z.open('train.csv')
df = pd.read_csv(train)

mapping = [{y:x} for x,y in enumerate(df.Category.unique())]
m_dict = {}
for m in mapping:
    m_dict.update(m)

df['c'] = df.Category.map(m_dict)
y_train = df['c'].values
X_train = df[['X', 'Y']].values

print(datetime.now())

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
max_depth=1, random_state=0).fit(X_train, y_train)

print(datetime.now())

s = clf.score(X_train, y_train)

print(s)
