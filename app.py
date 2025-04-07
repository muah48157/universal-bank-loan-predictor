from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/universal_bank_loan_predictor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Interpret result
    result = "Approved" if prediction[0] == 1 else "Not Approved"
    
    return render_template('index.html', prediction_text=f'Loan Status: {result}')

if __name__ == "__main__":
    app.run(debug=True)