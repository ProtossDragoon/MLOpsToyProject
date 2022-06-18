# source: https://www.kaggle.com/code/muleatharva/spam-ham-detection-system

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Convert text to numbers
from sklearn.feature_extraction.text import CountVectorizer 

# for mlflow
import os
from random import random, randint
import mlflow
import mlflow.sklearn

data = pd.read_csv('./spam.csv', encoding='latin-1')
data.dropna(how="any", inplace=True, axis=1)

# Change column names
data.columns = ['label', 'messages']
# data.messages[2]
# data.shape



# Check for any null value
data.isnull().sum()
data.info()
data.describe()
data.head()

len_ham = len(data[(data.label == 'ham')])
len_spam = len(data[(data.label == 'spam')])

# print("Ham Cases: ", len_ham)
# print("Spam Cases: ", len_spam)

arr = [len_ham, len_spam]
labels = ['Ham', 'Spam']

plt.pie(arr, labels=labels, explode=[0,0.2], shadow=True)
plt.show()

def text_preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'").replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not").replace("n't", " not").replace("what's", "what is").replace("it's", "it is").replace("'ve", " have").replace("i'm", "i am").replace("'re", " are").replace("he's", "he is").replace("she's", "she is").replace("'s", " own").replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ").replace("€", " euro ").replace("'ll", " will")
    return x  

data["Preprocessed Text"] = data["messages"].apply(lambda x: text_preprocess(x))
# data.head()

# print(data.messages[4])
# print(data["Preprocessed Text"][4])

data["label"] = data.label.map({'ham':0, 'spam':1})
# data.head()

X = data.messages
y= data.label

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=randint(1,100))

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train) 
predictions = naive_bayes.predict(testing_data)
accu_score = accuracy_score(y_test, predictions)

# model save
os.remove("./dbfs/my_project_models")
path = "./dbfs/my_project_models/"
if not os.path.exists(path):
    os.makedirs(path)

mlflow.sklearn.save_model(naive_bayes, modelpath)