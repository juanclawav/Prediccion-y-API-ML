from flask import Flask, request, jsonify
import pandas as pd
import os
from Predict import pipeline_lr, pipeline_dt, pipeline_rf

app = Flask(__name__)


@app.route('/predictLR', methods=['POST'])
def predictLR():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    prediction = pipeline_lr.predict(df)
    return jsonify({'prediction': prediction[0]})

@app.route('/predictRF', methods=['POST'])
def predictRF():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    prediction = pipeline_rf.predict(df)
    return jsonify({'prediction': prediction[0]})

@app.route('/predictDT', methods=['POST'])
def predictDT():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    prediction = pipeline_dt.predict(df)
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable or default to 5000
    app.run(host='0.0.0.0', port=port) 