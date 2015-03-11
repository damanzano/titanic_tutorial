'''
Created on 6/02/2015

@author: David Andres Manzano Herrera
'''
import csv as csv
import pandas as pd
import numpy as np
import pylab as P

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('csv/train.csv', header=0)
df_test = pd.read_csv('csv/test.csv', header=0)

df.info()

# Cleaning data

## 1. Convert different category fields into ints 
### Gender
### Train
df['Gender']= df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

### Test
df_test['Gender']= df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

### Embarked
### Train
# All missing Embarked -> just make them embark from most common place
if len(df.Embarked[ df.Embarked.isnull() ]) > 0:                        
    df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(df['Embarked'])))                                  # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }                                      # set up a dictionary in the form  Ports : index
df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)                 # Convert all Embark strings to int

### Test
if len(df_test.Embarked[ df_test.Embarked.isnull() ]) > 0:
    df_test.Embarked[ df_test.Embarked.isnull() ] = df_test.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
df_test.Embarked = df_test.Embarked.map( lambda x: Ports_dict[x]).astype(int)


## 2. Fill missing data
### Age
### Train
median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()

df['AgeFill'] = df['Age']

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]    
        
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

### Test
median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df_test[(df_test['Gender'] == i) & \
                              (df_test['Pclass'] == j+1)]['Age'].dropna().median()
                              
df_test['AgeFill'] = df_test['Age']

for i in range(0, 2):
    for j in range(0, 3):
        df_test.loc[ (df_test.Age.isnull()) & (df_test.Gender == i) & (df_test.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

df_test['AgeIsNull'] = pd.isnull(df_test.Age).astype(int)

### Fare
### Train There are no missing fares in train data

### Test
# All the missing Fares -> assume median of their respective class
if len(df_test.Fare[ df_test.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = df_test[ df_test.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        df_test.loc[ (df_test.Fare.isnull()) & (df_test.Pclass == f+1 ), 'Fare'] = median_fare[f]


## 3. Grouping and resume data
df['FamilySize'] = df['SibSp'] + df['Parch']
df_test['FamilySize'] = df['SibSp'] + df['Parch']
df['AgeClass'] = df.AgeFill * df.Pclass
df_test['AgeClass'] = df.AgeFill * df.Pclass

#df.FamilySize.hist()
#P.show()
#df.AgeClass.hist()
#P.show()

## 4. Dropping useless columns
df = df.drop(['Name', 'Sex', 'Age','Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1)

# Collect the test data's PassengerIds before dropping it
ids = df_test['PassengerId'].values
df_test = df_test.drop(['Name', 'Sex', 'Age','Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1)  

print(df.describe())                
print("---------------------------")
print(df_test.describe())  

# Get a numpy matrix from pandas df
train_data = df.values
test_data = df_test.values

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

print ('Training...')
# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

print ('Predicting...')
# Take the same decision trees and run it on the test data
output = forest.predict(test_data)
output = output.astype(np.int)

predictions_file = open("forestmodel.csv", "w", newline="")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print ('Done.')
