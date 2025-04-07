from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
model = joblib.load('model/universal_bank_loan_predictor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and convert to float
        features = [
            float(request.form['age']),
            float(request.form['experience']),
            float(request.form['income']),
            float(request.form['family']),
            float(request.form['ccavg']),
            float(request.form['education']),
            float(request.form.get('mortgage', 0)),  # Optional field
            1.0 if 'securities' in request.form else 0.0,
            1.0 if 'cd_account' in request.form else 0.0,
            1.0 if 'online' in request.form else 0.0,
            1.0 if 'creditcard' in request.form else 0.0
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        result = "Approved" if prediction == 1 else "Not Approved"
        
        return render_template('index.html', 
                            prediction_text=f'Loan Status: {result}',
                            show_result=True)
    
    except Exception as e:
        return render_template('index.html', 
                            prediction_text=f'Error: {str(e)}',
                            show_result=True)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
