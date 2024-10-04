import numpy as np 
import pandas as pd 
import sklearn 
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
reg = RandomForestClassifier()
g = pickle.load(open('Iris_project.pkl','rb'))
from flask import Flask , render_template , request

app = Flask(__name__)


@app.route("/")
def fun():
    return render_template("index.html")

@app.route("/sk", methods = ['POST'])
def predict():
    if request.method == 'POST':
        spl = request.form['sepal_length']
        spw = request.form['sepal_width']
        ptl = request.form['petal_length']
        ptw = request.form['petal_width']
        data = [[float(spl) , float(spw) , float(ptl) , float(ptw)]]
        outcome = g.predict(data)[0]
        if outcome == 0:
          return render_template("index.html" , prediction='setosa')
        elif outcome == 1:
            return render_template("index.html" , prediction='virsicolor')
        else:
            return render_template("index.html" , prediction='virginica')


if __name__ == "__main__":
    app.run(debug=True)
