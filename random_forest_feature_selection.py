'''
Created on 7/02/2015

@author: David Andres Manzano Herrera
'''

import csv as csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('csv/train.csv', header=0)
df_prod = pd.read_csv('csv/test.csv', header=0)

df.info()

# Cleaning data

## 1. Convert different category fields into ints 
### Gender
### Train
df['Gender']= df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

### Test
df_prod['Gender']= df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# ### Embarked
# ### Train
# # All missing Embarked -> just make them embark from most common place
# if len(df.Embarked[ df.Embarked.isnull() ]) > 0:                        
#     df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
# 
# Ports = list(enumerate(np.unique(df['Embarked'])))                                  # determine all values of Embarked,
# Ports_dict = { name : i for i, name in Ports }                                      # set up a dictionary in the form  Ports : index
# df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)                 # Convert all Embark strings to int
# 
# ### Test
# if len(df_prod.Embarked[ df_prod.Embarked.isnull() ]) > 0:
#     df_prod.Embarked[ df_prod.Embarked.isnull() ] = df_prod.Embarked.dropna().mode().values
# # Again convert all Embarked strings to int
# df_prod.Embarked = df_prod.Embarked.map( lambda x: Ports_dict[x]).astype(int)


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
        median_ages[i,j] = df_prod[(df_prod['Gender'] == i) & \
                              (df_prod['Pclass'] == j+1)]['Age'].dropna().median()
                              
df_prod['AgeFill'] = df_prod['Age']

for i in range(0, 2):
    for j in range(0, 3):
        df_prod.loc[ (df_prod.Age.isnull()) & (df_prod.Gender == i) & (df_prod.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

df_prod['AgeIsNull'] = pd.isnull(df_prod.Age).astype(int)

### Fare
### Train There are no missing fares in train data

### Test
# All the missing Fares -> assume median of their respective class
if len(df_prod.Fare[ df_prod.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = df_prod[ df_prod.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        df_prod.loc[ (df_prod.Fare.isnull()) & (df_prod.Pclass == f+1 ), 'Fare'] = median_fare[f]


## 3. Grouping and resume data
df['FamilySize'] = df['SibSp'] + df['Parch']
df_prod['FamilySize'] = df['SibSp'] + df['Parch']
df['AgeClass'] = df.AgeFill * df.Pclass
df_prod['AgeClass'] = df.AgeFill * df.Pclass

#df.FamilySize.hist()
#P.show()
#df.AgeClass.hist()
#P.show()

## 4. Dropping useless columns
df = df.drop(['Name', 'Sex', 'Age','Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1)

# Collect the test data's PassengerIds before dropping it
ids = df_prod['PassengerId'].values
df_prod = df_prod.drop(['Name', 'Sex', 'Age','Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1)  

print(df.describe())                
print("---------------------------")
print(df_prod.describe())  

# Get a numpy matrix from pandas df
train_data = df.values
x_train, x_test, y_train, y_test = cross_validation.train_test_split(train_data[0::,1::],train_data[0::,0]) 

prod_data = df_prod.values

print ('Training...')
# Create the forest object which will include all the parameters
# for the fit and  compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(x_train,y_train)
importances = forest.feature_importances_

# obtiene el vector de desviasiones estandar de las features
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(9):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    

data_headers = list(df.columns.values)[1:]
print (data_headers)
dheders=[data_headers[x] for x in indices]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(9), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(9), dheders)

plt.xlim([-1, 9])
plt.show()



print ('Predicting validation data...')
# Take the same decision trees and run it on the test data
y_pred = forest.predict(x_test)

print metrics.classification_report(y_test, y_pred, target_names=['Survived', 'Death'])

# Predict with production data
print ('Processing test data...')
y_pred = forest.predict(prod_data)
# Save to file
predictions_file = open("forestmodelevaluated_fselection.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, y_pred.astype(np.int)))
predictions_file.close()
print ('Done.')
