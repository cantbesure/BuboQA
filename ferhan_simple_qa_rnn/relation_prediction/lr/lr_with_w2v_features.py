
# coding: utf-8

# In[1]:

import os
import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics


# In[2]:

train_path = "./feature.train"
valid_path = "./feature.valid"
train_valid_path = "./feature.train_valid"
test_path = "./feature.test"


# ### Create training set (train + valid)

# In[10]:

X_train = []
y_train = []

with open(train_valid_path, 'r') as f:
    for line in f:
        items = line.split(" %%%% ")
        lineid = items[0].split("-")[0] + str(int(items[0].split("-")[1])+1)
        x = np.fromstring(items[1], sep=' ')

        X_train.append(x)
        y_train.append(items[2].strip())

print("done")


# In[11]:

X_train = np.array(X_train)
y_train = np.array(y_train)

print(X_train.shape)
print(y_train.shape)


# ### Create test set

# In[5]:

X_test = []
y_test = []

with open(test_path, 'r') as f:
    for line in f:
        items = line.split(" %%%% ")
        lineid = items[0].split("-")[0] + str(int(items[0].split("-")[1])+1)
        x = np.fromstring(items[1], sep=' ')

        X_test.append(x)
        y_test.append(items[2].strip())

print("done")


# In[6]:

X_test = np.array(X_test)
y_test = np.array(y_test)

print(X_test.shape)
print(y_test.shape)


# ### Run logistic regression - fit the model on (X_train, y_train)

# In[ ]:

clf = LogisticRegression()
clf.fit(X_train, y_train)
print("model trained!")


# In[ ]:

pickle.dump( clf, open( "lr_fit_train_valid.pkl", "wb" ) )


# In[14]:

predicted_train = clf.predict(X_train)
accuracy_on_train = 100.0 * np.mean(predicted_train == y_train)
print("accuracy_on_train: {}".format(accuracy_on_train))


# ### Test the LR model

# In[ ]:

predicted = clf.predict(X_test)
accuracy = 100.0 * np.mean(predicted == y_test)
print("accuracy on test dataset: {}".format(accuracy))


