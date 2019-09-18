import flask
from flask import request, jsonify
import json

import sys

import pandas as pd
import numpy

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

app = flask.Flask(__name__)
app.config["DEBUG"] = True

model_filename="model.pkl"

@app.route("/", methods=["GET"])
#Home page
def home():
	return '''<h1> Manny B in the house! Fraud Detection API : IOO</h1>
<p> A POC for the use of Machiasdfne learning to detect Credit Cards Frauds.</p>'''


@app.route("/api/v0/info", methods=['GET'])
#Sample GET function
def info():
    result = [
        {
            "Player" : "MannyB!",
            "Description" : "Deploying APIs all day! Everyday!",
        }
    ]
    return jsonify(result)


@app.route("/api/v0/predict", methods=["POST"])
def predict_transaction():
    vec=pd.DataFrame([request.json])
    model=joblib.load(model_filename)
    y_hat=str(model.predict(vec)[0])
    y_prob=str(model.predict_proba(vec)[0][1].round(3))
    
    result=[{"Prediction":y_hat,"Probability":y_prob}]
	
    return jsonify(result)

app.run()