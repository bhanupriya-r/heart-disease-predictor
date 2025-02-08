from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = load_model('hdp.keras')
df = pd.read_csv('processed_heart.csv')  # Load dataset
X_train = df.iloc[:, :-1].values
ss = StandardScaler()
ss.fit(X_train)

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
            print("Input Features:", features)
            # Convert into NumPy array & reshape for prediction
            features = np.array([features])  # Convert to NumPy array
            features_scaled = ss.transform(features)  # Shape (1, 12)

            # Make prediction
            prediction = model.predict(features_scaled)[0][0]   # Extract single value
            print("Model Prediction:", prediction)
            
            return render_template('result.html', prediction=prediction)

        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
