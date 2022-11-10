from flask import Flask, render_template,request
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
from sklearn import metrics
from sklearn.exceptions import NotFittedError
import pickle

with open('Sentiment_Analysis_LR.pickle' , 'rb') as f:
    lr = pickle.load(f)

data=["Good movie"]
#print(type(data))
T=vect.transform(data)
res=lr.predict(T)
print(res)