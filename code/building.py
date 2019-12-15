import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

"""## Converting dummies"""

# Import dataset
df = pd.read_csv('mammographic_masses.csv')
df.head()

"""## Converting dummies"""

df = pd.get_dummies(df)
df.head()

#Rearranging the dataframe

masses_data = df[['BI-RADS', 
                           'age', 
                           'shape', 
                           'margin', 
                           'density',
                           'androgen_history',
                           'prev_visit',
                           'blood_group_A', 
                           'blood_group_AB',
                           'blood_group_B',
                           'blood_group_O',
                           'severity']]



df = masses_data
df.columns

"""## Checking for outliers"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
fig= plt.figure(figsize=(20,6))

plt.subplot(2,5,1)
plt.title("BI-RADS")
plt.hist(list(df['BI-RADS']))

plt.subplot(2,5,2)
plt.title("age")
plt.hist(list(df['age']))

plt.subplot(2,5,3)
plt.title("shape")
plt.hist(list(df['shape']))

plt.subplot(2,5,4)
plt.title("margin")
plt.hist(list(df['margin']))

plt.subplot(2,5,5)
plt.title("density")
plt.hist(list(df['density']))

plt.subplot(2,5,6)
plt.title("androgen_history")
plt.hist(list(df['androgen_history']))

plt.subplot(2,5,7)
plt.title("prev_visit")
plt.hist(list(df['prev_visit']))

plt.subplot(2,5,8)
plt.title("severity")
plt.hist(list(df['severity']))

plt.show()

"""## Missing data"""

data = df.copy()

imputer = Imputer(missing_values = np.NAN, strategy='median')
data.iloc[:,:] = imputer.fit_transform(data.iloc[:,:])
data.isnull().sum()

"""## Different types of data"""

data.dtypes

"""## Irrelavant features"""

del data['BI-RADS']
data.head()

"""## Splitting into Train and Test"""

data.to_csv(r'data.csv')

y = data.loc[:,'severity'].values
X = data.loc[:,data.columns!='severity'].values

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state=0)
print("Training set: ", X_train.shape, y_train.shape)
print("Test set: ", X_test.shape, y_test.shape)

"""## Feature Engineering"""

#Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""## Model Building & Evaluation



"""### 1. Supervised vector machine"""

# Before Hyper-parameter optimization
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, model.predict(X_test)))
print(" Model Accuracy",model.score(X_test, y_test))

from sklearn.model_selection import GridSearchCV
parameters= {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
model = SVC()
model_cv=GridSearchCV(model, param_grid= parameters,cv=10)
model_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",model_cv.best_params_)
print("accuracy :",model_cv.best_score_)

# After Hyper-parameter optimization
model = SVM()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, model.predict(X_test)))
print(" Model Accuracy",model.score(X_test, y_test))
get_learning_curve(model,"SVM")

# Stratified K Fold CV

from sklearn.model_selection import KFold
model = SVM()model.fit(X_train, y_train)
array = [[0,0],[0,0]]
scores = []
cv = KFold(n_splits = 10, random_state=42, shuffle = False)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores.append(model.score(X_test, y_test))
    c = confusion_matrix(y_test, model.predict(X_test))
    array = array + c
cm = pd.DataFrame(array, index = ['Benign', 'Malignant'], columns = ['Benign', 'Malignant'])

# Print Results and Put the results into the dataframe
  cm = get_results(cm,scores)
  acc, pre, rec, f1 = get_scores(cm)
  compare.loc[len(compare)] = ["SVM", round(acc,2), round(pre,2), round(rec,2), round(f1,2)]
  compare.head()

"""### 2. Decision Tree Classifier"""

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, model.predict(X_test)))
print(" Model Accuracy",model.score(X_test, y_test))

"""### How gridsearch works


1.   Try every combination of your parameter grid
2.   For each of them it will do a K-fold cross validation - By default CV = 3

1.   Select the best available.
"""

from sklearn.model_selection import GridSearchCV
parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
model = DecisionTreeClassifier()
model_cv=GridSearchCV(model, param_grid=parameters,cv=10)
model_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",model_cv.best_params_)
print("accuracy :",model_cv.best_score_)

model = DecisionTreeClassifier(max_depth=19, min_samples_split=50)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, model.predict(X_test)))
print(" Model Accuracy",model.score(X_test, y_test))
get_learning_curve(model,"Decision Tree")

# Stratified K Fold CV
## Go through uniform and stratified k fold.
from sklearn.model_selection import KFold
model = DecisionTreeClassifier(max_depth=19, min_samples_split=50)
model.fit(X_train, y_train)
array = [[0,0],[0,0]]
scores = []
cv = KFold(n_splits = 10, random_state=42, shuffle = False)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores.append(model.score(X_test, y_test))
    c = confusion_matrix(y_test, model.predict(X_test))
    array = array + c
cm = pd.DataFrame(array, index = ['Benign', 'Malignant'], columns = ['Benign', 'Malignant'])

# Print Results and Put the results into the dataframe
  cm = get_results(cm,scores)
  acc, pre, rec, f1 = get_scores(cm)
  compare.loc[len(compare)] = ["Decision Tree", round(acc,2), round(pre,2), round(rec,2), round(f1,2)]
  compare.head()

"""### 3. Random forest classifier"""

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, model.predict(X_test)))
print(" Model Accuracy",model.score(X_test, y_test))



from sklearn.model_selection import GridSearchCV
parameters = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
model = RandomForestClassifier()
model_cv=GridSearchCV(model, param_grid=parameters,cv=10)
model_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",model_cv.best_params_)
print("accuracy :",model_cv.best_score_)

model = RandomForestClassifier(criterion="gini", max_depth = 8, max_features="auto", n_estimators=200)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, model.predict(X_test)))
print(" Model Accuracy",model.score(X_test, y_test))
get_learning_curve(model,"Random Forest")

# Stratified K Fold CV

from sklearn.model_selection import KFold
model = model = RandomForestClassifier(criterion="gini", max_depth = 8, max_features="auto", n_estimators=200)
model.fit(X_train, y_train)
array = [[0,0],[0,0]]
scores = []
cv = KFold(n_splits = 10, random_state=42, shuffle = False)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores.append(model.score(X_test, y_test))
    c = confusion_matrix(y_test, model.predict(X_test))
    array = array + c
cm = pd.DataFrame(array, index = ['Benign', 'Malignant'], columns = ['Benign', 'Malignant'])

# Print Results and Put the results into the dataframe
  cm = get_results(cm,scores)
  acc, pre, rec, f1 = get_scores(cm)
  compare.loc[len(compare)] = ["Random Forest", round(acc,2), round(pre,2), round(rec,2), round(f1,2)]
  compare.head()

"""### 4. XGBoost Classifier"""

#Without KFold Cross validation 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, model.predict(X_test)))
print(" Model Accuracy",model.score(X_test, y_test))

#Without KFold Cross validation with hyper-parameter optimization
from sklearn.model_selection import GridSearchCV
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

model= XGBClassifier()
model_cv=GridSearchCV(model, param_grid=parameters,cv=10)
model_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",model_cv.best_params_)
print("accuracy :",model_cv.best_score_)

model = XGBClassifier(algorithm= 'auto', leaf_size=1, n_jobs=-1, n_neighbors= 4)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, model.predict(X_test)))
print(" Model Accuracy",model.score(X_test, y_test))
get_learning_curve(model,"XG Boost")



# Stratified K Fold CV

from sklearn.model_selection import KFold
model = XGBClassifier(algorithm= 'auto', leaf_size=1, n_jobs=-1, n_neighbors= 4)
model.fit(X_train, y_train)
array = [[0,0],[0,0]]
scores = []
cv = KFold(n_splits = 10, random_state=42, shuffle = False)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores.append(model.score(X_test, y_test))
    c = confusion_matrix(y_test, model.predict(X_test))
    array = array + c
cm = pd.DataFrame(array, index = ['Benign', 'Malignant'], columns = ['Benign', 'Malignant'])

# Print Results and Put the results into the dataframe
  cm = get_results(cm,scores)
  acc, pre, rec, f1 = get_scores(cm)
  compare.loc[len(compare)] = ["XG Boost", round(acc,2), round(pre,2), round(rec,2), round(f1,2)]
  compare.head()

"""### 5. Naive Bayes Classifier"""

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
model = GaussianNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, model.predict(X_test)))
print(" Model Accuracy",model.score(X_test, y_test))
get_learning_curve(model,"Naive Bayes")

#There isn't a hyper-parameter to tune, so you have nothing to grid search over.

# Stratified K Fold CV

from sklearn.model_selection import KFold
model = GaussianNB()
model.fit(X_train, y_train)
array = [[0,0],[0,0]]
scores = []
cv = KFold(n_splits = 10, random_state=42, shuffle = False)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores.append(model.score(X_test, y_test))
    c = confusion_matrix(y_test, model.predict(X_test))
    array = array + c
cm = pd.DataFrame(array, index = ['Benign', 'Malignant'], columns = ['Benign', 'Malignant'])

# Print Results and Put the results into the dataframe
  cm = get_results(cm,scores)
  acc, pre, rec, f1 = get_scores(cm)
  compare.loc[len(compare)] = ["Naive Bayes", round(acc,2), round(pre,2), round(rec,2), round(f1,2)]
  compare.head()

"""### 6. Nearest Neighbout classifier"""

#Without KFold Cross validation 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

model = KNeighborsClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, model.predict(X_test)))
print(" Model Accuracy",model.score(X_test, y_test))

#Without KFold Cross validation with hyper-parameter optimization
from sklearn.model_selection import GridSearchCV
#k_range = list(range(1,31))
parameters = {'n_neighbors':[4,5,6,7],
              'leaf_size':[1,3,5],
              'algorithm':['auto', 'kd_tree'],
              'n_jobs':[-1]}

model= KNeighborsClassifier()
model_cv=GridSearchCV(model, param_grid=parameters,cv=10)
model_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",model_cv.best_params_)
print("accuracy :",model_cv.best_score_)

# After Optimization
model = KNeighborsClassifier(algorithm="auto", leaf_size=1, n_jobs=-1, n_neighbors= 7 )
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, model.predict(X_test)))
print(" Model Accuracy",model.score(X_test, y_test))
get_learning_curve(model,"KNN")

# Stratified K Fold CV

from sklearn.model_selection import KFold
model = KNeighborsClassifier(algorithm="auto", leaf_size=1, n_jobs=-1, n_neighbors= 7 )
model.fit(X_train, y_train)
array = [[0,0],[0,0]]
scores = []
cv = KFold(n_splits = 10, random_state=42, shuffle = False)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores.append(model.score(X_test, y_test))
    c = confusion_matrix(y_test, model.predict(X_test))
    array = array + c
cm = pd.DataFrame(array, index = ['Benign', 'Malignant'], columns = ['Benign', 'Malignant'])

# Print Results and Put the results into the dataframe
  cm = get_results(cm,scores)
  acc, pre, rec, f1 = get_scores(cm)
  compare.loc[len(compare)] = ["KNN", round(acc,2), round(pre,2), round(rec,2), round(f1,2)]
  compare.head()

"""### 7. Logistic Regression"""

#Without KFold Cross validation 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, model.predict(X_test)))
print(" Modeel Accuracy",model.score(X_test, y_test))

#Without KFold Cross validation with hyper-parameter optimization
from sklearn.model_selection import GridSearchCV
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
model=LogisticRegression()
model_cv=GridSearchCV(model,grid,cv=10)
model_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",model_cv.best_params_)
print("accuracy :",model_cv.best_score_)

# After Optimization
model = LogisticRegression(C=1, penalty= "l2")
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, model.predict(X_test)))
print(" Model Accuracy",model.score(X_test, y_test))
get_learning_curve(model,"Logistic Regression")

# Stratified K Fold CV

from sklearn.model_selection import KFold
model = LogisticRegression(C=1, penalty= "l2")
model.fit(X_train, y_train)
array = [[0,0],[0,0]]
scores = []
cv = KFold(n_splits = 10, random_state=42, shuffle = False)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores.append(model.score(X_test, y_test))
    c = confusion_matrix(y_test, model.predict(X_test))
    array = array + c
cm = pd.DataFrame(array, index = ['Benign', 'Malignant'], columns = ['Benign', 'Malignant'])

# Print Results and Put the results into the dataframe
  cm = get_results(cm,scores)
  acc, pre, rec, f1 = get_scores(cm)
  compare.loc[len(compare)] = ["Logistic", round(acc,2), round(pre,2), round(rec,2), round(f1,2)]
  compare.head()
