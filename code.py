# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
#from keras.utils.np_utils import to_categorical
# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train.drop(["Name", "Cabin", "Embarked", "PassengerId"], axis = 1)
test1 = test.drop(["Name", "Cabin", "Embarked", "PassengerId"], axis = 1)
test = test.drop(["Name", "Cabin", "Embarked", "PassengerId"], axis = 1).values
X1 = train.drop(["Survived"], axis = 1)
X = train.drop(["Survived"], axis = 1).values
y = train.iloc[:, 0].values

g = sns.countplot(y) #Dsiplays the frequency of occurence of the values in Y_train in the form of a bar graphs

y.value_counts() #Dsiplays the frequency of occurence of the values in Y_train

#Combining the Data
frames = [X1, test1]
result = pd.concat(frames ,keys = ['X','test'])

#Handling Missing Data
#X1.isnull().describe()
#test1.isnull().describe()
result = result.values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(result[:, 2:3])
result[:, 2:3] = imputer.transform(result[:, 2:3])
imputer = imputer.fit(result[:, -1:])
result[:, -1:] = imputer.transform(result[:, -1:])


#Encoding
#X["Sex"] = X["Sex"].map( {'female': 1, 'male': 0} ).astype(int)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
result[:,5] = labelencoder.fit_transform(result[:,5])
result[:,1] = labelencoder.fit_transform(result[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])
result = onehotencoder.fit_transform(result).toarray()
onehotencoder = OneHotEncoder(categorical_features = [5])
result = onehotencoder.fit_transform(result).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
result = sc.fit_transform(result)

X1 = pd.DataFrame(X)
test1 = pd.DataFrame(test)
#Splitting the combined dataframe back to the originial dataframes
result1 = pd.DataFrame(result)
X1 = result1.iloc[0:891, :].values
test1 = result1.iloc[891:, :].values 

#Splitting into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X1, y, test_size=0.2, random_state = 0)


#from xgboost import XGBClassifier
classifier = XGBClassifier(C = 1, colsample_bytree = 0.7, eta = 0.01, gamma = 0.3, subsample = 0.5)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Confusion Matrix
confusion_mtx = confusion_matrix(y_test, y_pred) 
sns.heatmap(confusion_mtx, annot=True, fmt='d')

#Checking score (Accuracy)
classifier.score(X_test,y_test)

#10-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

#Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [
{'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.3, 0.5, 0.7, 0.9], 'eta':[0.01, 0.1,  0.2 ,0.3], 'subsample':[0.5, 0.6, 0.7, 0.8], 'colsample_bytree': [0.5,  0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["auc","error"]
classifier.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Creating Submission File
final = classifier.predict(test1)
final = pd.DataFrame(final)
final['PassengerId'] = pd.Series(data = np.arange(892,1310), index=final.index)
final.columns = ['Survived','PassengerId']
columnsTitles=["PassengerId","Survived"]
final=final.reindex(columns=columnsTitles)

#Exporting the dataframe
final.to_csv('Predictions.csv', index = False)

#Trying out ANN now
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense,LeakyReLU
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
#No of hidden nodes
#Nh=Ns/(α∗(Ni+No))
#Ni  = number of input neurons.
#No = number of output neurons.
#Ns = number of samples in training data set.
#α = an arbitrary scaling factor usually 2-10.
Nh = int(891/32)
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = Nh, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
#classifier.add(LeakyReLU(alpha=0.1))
classifier.add(Dropout(0.01))

# Adding the second hidden layer
classifier.add(Dense(units = Nh, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(LeakyReLU(alpha=0.1))
classifier.add(Dropout(0.01))

## Adding the third hidden layer
#classifier.add(Dense(units = Nh, kernel_initializer = 'uniform', activation = 'relu'))
##classifier.add(LeakyReLU(alpha=0.1))
#classifier.add(Dropout(0.01))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dropout(0.01))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
classifier.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
# Fitting the ANN to the Training set
history = classifier.fit(X1, y, batch_size = 25, epochs = 10000, callbacks = [learning_rate_reduction])

# Part 3 - Making predictions and evaluating the model

confusion_mtx = confusion_matrix(y_test, final) 
# plot the confusion matrix 
#import seaborn as sns
sns.heatmap(confusion_mtx, annot=True, fmt='d')
# Predicting the Test set results
final = classifier.predict(test1)

final = (final > 0.5)
final = final.astype(int)
final = pd.DataFrame(final)
final['PassengerId'] = pd.Series(data = np.arange(892,1310), index=final.index)
final.columns = ['Survived','PassengerId']
columnsTitles=["PassengerId","Survived"]
final=final.reindex(columns=columnsTitles)

#Exporting the dataframe
final.to_csv('Predictions_ANN.csv', index = False)

#Plotting the curves for error and accuracy
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)