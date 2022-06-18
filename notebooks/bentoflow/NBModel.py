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

class NBModel():
    def __init__(self):
        self.DATA_PATH = '/home/nyam/MLOpsToyProject/notebooks/bentoflow/spam.csv'
        self.count_vector = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.training_data = None

    def textPreprocess(self,x):
        x = str(x).lower()
        x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'").replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not").replace("n't", " not").replace("what's", "what is").replace("it's", "it is").replace("'ve", " have").replace("i'm", "i am").replace("'re", " are").replace("he's", "he is").replace("she's", "she is").replace("'s", " own").replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ").replace("€", " euro ").replace("'ll", " will")
        return x  

    def getCountVector(self):
        data = pd.read_csv(self.DATA_PATH, encoding='latin-1')
        data.dropna(how="any", inplace=True, axis=1)
        data.columns = ['label', 'messages']

        data["messages"].apply(lambda x: self.textPreprocess(x))
        data["label"] = data.label.map({'ham':0, 'spam':1})

        X = data.messages
        y= data.label

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=1) #random_state=randint(1,100))

        # Instantiate the CountVectorizer method
        self.count_vector = CountVectorizer()

        # Fit the training data and then return the matrix
        self.training_data = self.count_vector.fit_transform(self.X_train)
        return self.count_vector

    def modelSave(self):
        naive_bayes = MultinomialNB()
        naive_bayes.fit(self.training_data, self.y_train) 

        # model save
        path = "./dbfs/my_project_models/"
        if not os.path.exists(path):
            os.makedirs(path)

        modelpath = "./dbfs/my_project_models/"
        mlflow.sklearn.save_model(naive_bayes, modelpath)
