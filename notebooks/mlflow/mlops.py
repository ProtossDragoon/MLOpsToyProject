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
from mlflow import log_metric, log_metrics, log_param, log_artifacts


# for iter random seed
from sys import argv # seed == argv[1]
seed = int(argv[1])

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

log_param('random_seed',seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
# print("No. of rows in total set: ", data.shape[0])
# print("No. of rows in training set: ", X_train.shape[0])
# print("No. of rows in testing set: ", X_test.shape[0])

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train) 

predictions = naive_bayes.predict(testing_data)

# print('Accuracy score ', format(accuracy_score(y_test, predictions))) 
# print('Precision score', format(precision_score(y_test, predictions))) 
# print('Recall score   ', format(recall_score(y_test, predictions))) 
# print('F1 score   ', format(f1_score(y_test, predictions))) 

accu_score = accuracy_score(y_test, predictions)
prec_score = precision_score(y_test, predictions)
rcal_score = recall_score(y_test, predictions)
fone_score = f1_score(y_test, predictions)

log_metric('Accuracy' ,accu_score)
log_metric('Precision',prec_score)
log_metric('Recall'   ,rcal_score)
log_metric('F1'       ,fone_score)

path = "outputs"
if not os.path.exists(path):
    os.makedirs(path)
with open("%s/test.txt"%(path),'a') as f:
    f.write("mlflow test - " + str(seed))
log_artifacts("%s"%(path))

# model save
path = "./dbfs/my_project_models/"
if not os.path.exists(path):
    os.makedirs(path)

modelpath = "./dbfs/my_project_models/model-%f" % (accu_score)
if not os.path.exists(modelpath):
    mlflow.sklearn.save_model(naive_bayes, modelpath)

test_arr = ["This is the 3rd time we have tried to contact Jack",
"Dear Investor,With reference to NSE circular NSE/INSP/46704 dated December 17, 2020 and NSE/INSP/46960 dated January 08, 2021, Stock Brokers are required to upload clients fund balance and securities balance on weekly basis.",
"pleaes register now",
"Get Dad an eco-friendly gift he will love this Father's Day! Make it meaningful by adding your own custom saying. Our durable Octaroma comes in 5 awesome colors, fits in your car's cup holder, includes a drinkable spill-proof lid and keeps drinks hot or cold for hours.",
"Get the engraving for FREE with any past Wacaco coffee / espresso machine purchase that included a registration card. Simply add your Octaroma with personalized message to your cart, then enter the serial number from your registration card as a promo code at checkout to receive your FREE engraving."
]

# Testing the Model
mail = []
for tc in test_arr:
    trial = pd.Series(tc)
    test = count_vector.transform(trial)
    mail.append(naive_bayes.predict(test))
# log_metric('ham',int(mail[0]))
# log_metric('spam_1',int(mail[1]))
# log_metric('spam_2',int(mail[2]))
# log_metric('spam_3',int(mail[3]))
# log_metric('spam_4',int(mail[4]))
log_metrics({'ham': int(mail[0]), 'spams': int(mail[1])*1000 + int(mail[2])*100 + int(mail[3])*10 + int(mail[4])})
# # for test load model
# import joblib 

# mail = []
# for tc in test_arr:
#     trial = pd.Series(tc)
#     test = count_vector.transform(trial)
#     mail.append(naive_bayes.predict(test))
    
# print('ham    ' + str(mail[0]))
# print('spam_1 ' + str(mail[1]))
# print('spam_2 ' + str(mail[2]))
# print('spam_3 ' + str(mail[3]))
# print('spam_4 ' + str(mail[4]))

# mail = []
# file_name = modelpath+'/model.pkl'
# naive_bayes_loaded = joblib.load(file_name) 

# for tc in test_arr:
#     trial = pd.Series(tc)
#     test = count_vector.transform(trial)
#     mail.append(naive_bayes_loaded.predict(test))
    
# print('ham    ' + str(mail[0]))
# print('spam_1 ' + str(mail[1]))
# print('spam_2 ' + str(mail[2]))
# print('spam_3 ' + str(mail[3]))
# print('spam_4 ' + str(mail[4]))