#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-19T10:56:51.683Z
"""

# ### Spam Mail Prediction Using Machine Learning


#spam mail:free entry to competetion to call this no'
    
#ham mail:you are slected for game from relatives thank you"

#vectorizer - purpose of import is convert text data mail data to numerical values(meaningful nos)
#to understand our ML model - convert text to feature vectors

#import logistic regression to classify spam or ham mail

#accuracy score: To train data to LR model / to testdata to evaluate our model thats why imported 
#how well it is performing


# ### Importing dependencies


#importing dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ### Data Collection and Preprocessing


#loading csv data file to pandas data frame

raw_mail_data = pd.read_csv('mail_data.csv')
raw_mail_data.head()

#this ds have missing values, null values

#replacce all null values into null strings

mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')


#printing all the 5 rows of the Data frame
mail_data.head()

#checking the rows and no of colms in dataframe 
mail_data.shape

#label encoding - convert ham,spam variables into numerical values
#for spam mail is 0 and ham mail is 1

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1 #code will cahnge all spam mail as 0 and ham as 1

# ##### spam - 0
# 
# ##### ham - 1


#seperating the data as texts and label/

x=mail_data['Message']

y=mail_data['Category']

print(x)

print(y)

# x_train y_train - training data
# x_test, y_test - testing data

#splitting the data into train and test to evaluate the model
'''

random state: parameter used to control randomness in algorithms so that 
you get consistent and reproducible results every time you run your code.

''' 
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=3)


print(x.shape)
print(x_train.shape) # 80% of data training
print(x_test.shape)  # 20% of data for testing



# #### Feature Extraction:


#transform the text data to feature vectors that can be used as input to ML model (LR)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

x_train_features = feature_extraction.fit_transform(x_train)

x_test_features = feature_extraction.transform(x_test)   

#convert y_test, t_train values as integers

y_train = y_train.astype('int') 
y_test = y_test.astype('int')

print(x_train_features)

# #### Training the Machine learning model


# model training - Logistic Regression

model = LogisticRegression()

#training the logistic regression model with training data

model.fit(x_train_features,y_train)



# #### Evaluating the Model trained Model


# Predict the train Data

prediction_on_training_data = model.predict(x_train_features)
accuracy_on_train_data = accuracy_score(y_train, prediction_on_training_data)

print("Accuracy :",accuracy_on_train_data)

 # Predict the test Data

prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)

print("Accuracy :",accuracy_on_test_data)



# ### Building a Predictive System


input_mail = ["07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow"]

#convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

#making prediction
prediction = model.predict(input_data_features)
if(prediction == 0):
    print("Spam Mail")
else:
    print("Ham Mail")