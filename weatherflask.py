from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Get the absolute path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to the model file
model_path = os.path.join(script_dir, 'model.pkl')

# Load the model using joblib
model = joblib.load(model_path)

def preprocess_input(precipitation, temp_max, temp_min, wind):
    input_data = pd.DataFrame({
        'precipitation': [float(precipitation)],
        'temp_max': [float(temp_max)],
        'temp_min': [float(temp_min)],
        'wind': [float(wind)]
    })

    return input_data

def predict_weather(input_data):
    prediction = model.predict(input_data)
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        precipitation = request.form['precipitation']
        temp_max = request.form['temp_max']
        temp_min = request.form['temp_min']
        wind = request.form['wind']

        input_data = preprocess_input(precipitation, temp_max, temp_min, wind)
        prediction = predict_weather(input_data)

        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
