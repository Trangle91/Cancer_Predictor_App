import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

def convert(data):
    temp = data[6]
    tempList = [ data[0], data[1], data[2], data[3], data[4], data[5], 0.0, 0.0, 0.0, 0.0]
    if temp == 1:
        tempList[6] = 1.0
    elif temp == 2:
        tempList[7] = 1.0
    elif temp == 3:
        tempList[8] = 1.0
    elif temp == 4:
        tempList[9] = 1.0
    return tempList
        
@app.route('/predict',methods=['POST'])
def predict():
    result = ""
    int_features = [float(x) for x in request.form.values()]
     
    int_features = convert(int_features)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction[0]:
        result = "Malignant"
    else:
        result = "Benign"
    return render_template('index.html', prediction_text='The tumor is {}'.format(result))


if __name__ == "__main__":
    app.run(debug=True)