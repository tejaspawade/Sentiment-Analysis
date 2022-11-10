from flask import Flask, render_template,request
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
from sklearn import metrics
import pickle
import csv
app = Flask(__name__)

sentiments=["Negative","Some what negative","Neutral","Some what positive","Positive"]

with open('logistic_regression_pickle' , 'rb') as f:
    lr = pickle.load(f)

@app.route("/")
def hello_world():
    return render_template('sent_1.html')

@app.route('/submit',methods=["GET","POST"])
def submit():
    if request.method=='POST':
        data=request.form['inp']

        res=(lr.predict([data]))
        with open('Movie_sentiments.csv', 'a', newline="") as file:
            myFile = csv.writer(file)

            #myFile.writerow(["Sentiment","Label"])
            myFile.writerow([data,sentiments[(int(res))]])
    return "Sentiment is "+ sentiments[(int(res))]
    

if __name__ == '__main__':
    app.run(debug=True)