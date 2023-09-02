# pip instal flask
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('pipe.pkl')

# 1. Front end - Display the html files
# 2. Back end - route the requests and put the values


@app.route('/')
def landing_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def prediction():
    test_df = pd.DataFrame([request.form])
    # print(test_df)
    value = model.predict(test_df)[0]
    value = str(round(value, 2))
    return render_template('predict.html', price=value + "Lakhs")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
