'''
Created on 24/01/2015

@author: David Andres Manzano Herrera
'''
import pandas as pd
import numpy as np

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('csv/train.csv', header=0)

#print df.head(3)
#print type(df)
#print df.dtypes
print df.info()
print df.describe()

print df.Age.mean()

#print df[df.Age > 60][['Sex', 'Pclass', 'Age', 'Survived']]
print df[df.Age.isnull()][['Sex', 'Pclass', 'Age']]

#for i in range(1,4):
#    print i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ])

for i in range(1,4):
    print i, len(df[(df.Sex == 'male') & (df.Pclass == i) ])