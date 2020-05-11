from flask import Flask,render_template,request
import joblib
import numpy as np


app=Flask(__name__)

ss=joblib.load("ss.pkl")
model=joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method =="POST":
        CRIM = float(request.form['CRIM'])
        ZN = float(request.form['ZN'])
        INDUS = float(request.form['INDUS'])
        NOX = float(request.form['NOX'])
        RM = float(request.form['RM'])
        AGE = float(request.form['AGE'])
        RAD = float(request.form['RAD'])
        PTRATIO = float(request.form['PTRATIO'])
        B = float(request.form['B'])
        LSTAT = float(request.form['LSTAT'])
        pred_args = [CRIM, ZN, INDUS, NOX,RM,AGE,RAD,PTRATIO,B,LSTAT]
        arr=np.array(pred_args)
        arr = arr.reshape(1,-1)
        arr=ss.transform(arr)
        prediction=model.predict(arr)
    return render_template('predict.html',prediction=prediction)    
        







if __name__=="__main__":
    app.run()
