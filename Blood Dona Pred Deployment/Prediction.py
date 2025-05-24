# Importing main Libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

# Creating Flask app
app = Flask(__name__)

# Loading model and preprocessor
model = pickle.load(open('blood_donation_model.pkl', 'rb'))
preprocessor = pickle.load(open('blood_donation_preprocessor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('tam.html', prediction=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Getting the form data from the request object
        unnamed_0 = int(request.form.get('unnamed_0'))
        months_since_last_donation = int(request.form.get('months_since_last_donation'))
        number_of_donations = int(request.form.get('number_of_donations'))
        total_volume_donated = int(request.form.get('total_volume_donated'))
        months_since_first_donation = int(request.form.get('months_since_first_donation'))
        recency_of_donations = float(request.form.get('recency_of_donations'))
        donation_frequency = float(request.form.get('donation_frequency'))
        donation_history_interaction = int(request.form.get('donation_history_interaction'))
        donation_consistency = float(request.form.get('donation_consistency'))
        relative_recency = int(request.form.get('relative_recency'))
        log_total_volume_donated = float(request.form.get('log_total_volume_donated'))

        # Create a DataFrame with the same column names used during training
        feature_dict = {
            'Unnamed: 0': unnamed_0,
            'Months since Last Donation': months_since_last_donation,
            'Number of Donations': number_of_donations,
            'Total Volume Donated (c.c.)': total_volume_donated,
            'Months since First Donation': months_since_first_donation,
            'Recency of Donations': recency_of_donations,
            'Donation Frequency': donation_frequency,
            'Donation History Interaction': donation_history_interaction,
            'Donation Consistency': donation_consistency,
            'Relative Recency': relative_recency,
            'Log Total Volume Donated': log_total_volume_donated
        }

        # Convert the dictionary into a DataFrame
        features_df = pd.DataFrame([feature_dict])

        # Preprocessing - Apply the preprocessor (e.g., scaling, feature transformations)
        feature_sqrt = np.sqrt(features_df)
        feature_scaled = preprocessor.transform(feature_sqrt)
        
        # Making the prediction
        prediction = model.predict_proba(feature_scaled)[:, 1]

        return render_template('tam.html', prediction=prediction[0])

    except Exception as e:
        return render_template('tam.html', prediction=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
