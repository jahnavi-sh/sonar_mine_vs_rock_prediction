#This algorithm is about predicting SONAR rocks against mines with the help of machine learning.
#It is especially useful in naval warfare where it is important 
# for submarines to predict whether the object below it is a 
# mine or a rock. The project is to train an algorithm to 
# discriminate between sonar signals bounced off a mine and 
# those bounced off a rock.  

#work flow - 
#1. collect sonar data (data collected from experiment done in laboratory setting)
#2. data preprocessing 
#3. train test split 
#4. model training- logistic regression model 

#load libraries 

#linear algebra
import numpy as np          #construct matrices

#data processing 
import pandas as pd         #data preprocessing and exploration

#algorithms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading dataset to pandas dataframe 
#pandas dataframe is a 2D tabular structure with rows and columns.
sonar_data = pd.read_csv(r"C:\Users\jahna\Downloads\sonar_data.csv", header=None)
 
#data preprocessing 
#to understand the data, we first need to view the data

sonar_data.head()
#this shows the first five rows of the dataset
#we can see that there are 5 rows and 61 columns of data
#the 60th column gives us the rock and mine classification label
#R - the object is rock 
#M - the object is mine

#view number of rows and columns 
sonar_data.shape
#we can see the 60 columns are the features and 61th column is output (categorical value)

#to better understand the numeric data, view statistical parameters 
#this gives us an understanding of the central tendencies of data
sonar_data.describe()

#check class balance both in forms of figure and plot
#view count of mines and rocks data 
sonar_data[60].value_counts()
#we see it's a balanced data 

#group data into mines and rocks 
sonar_data.groupby(60).mean()

#separating data and label 
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

#training and testing data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

print (X.shape, X_train.shape, X_test.shape)

#model training 
model = LogisticRegression()
model.fit(X_train, Y_train)

#evaluate model 

#accuracy of training data 
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("accuracy on training data", training_data_accuracy)

#accuracy score on test data 
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("accuracy on test data", test_data_accuracy)

#we can see that that the accuracy on training data is 83% and 
#accuracy on test data is 76% which is a very good score 
#usually a score of more than 70% is considered good. 

#make a prediction system 
input_data=() 

#chaning the input data to numpy array 
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for one instance 
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshape)
print (prediction)

if(prediction[0]=='R'):
    print("the object is a rock ")
else:
    print ("the object is a mine")