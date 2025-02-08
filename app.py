from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('hdp.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract input features from form
            features = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                float(request.form['trestbps']),
                float(request.form['chol']),
                float(request.form['fbs']),
                float(request.form['restecg']),
                float(request.form['thalach']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca'])
            ]

            # Convert into NumPy array & reshape for prediction
            data = np.array([features])  # Shape (1, 12)

            # Make prediction
            prediction = model.predict(data)[0][0]  # Extract single value

            return render_template('result.html', prediction=prediction)

        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
