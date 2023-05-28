from flask import Flask, render_template, request
import os
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'file_diabetes.pkl'
#model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename)
#model = joblib.load(filename)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    Pregnancies = float(request.form['Pregnancies'])
    Glucose = float(request.form['Glucose'])
    BMI = float(request.form['BMI'])
    Age = float(request.form['Age'])



    pred = model.predict(np.array([[Pregnancies, Glucose, BMI, Age ]]))
    print(pred)
    return render_template('index.html', predict=str(pred))


if __name__ == '__main__':
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=port)
