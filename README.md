# AgriAssist

A Python-based tool designed to assist farmers and customers in agriculture. For farmers, it predicts crop health, provides actionable farming suggestions, and calculates profits. For customers, it ensures fair pricing and savings by connecting them with farmers.

## Features

- **Farmer Mode:**
  - Predicts crop health (Good/Bad) using Random Forest AI.
  - Provides irrigation, pesticide, and soil fertility suggestions.
  - Calculates fair sale price and estimated profit per acre.
- **Customer Mode:**
  - Estimates fair price based on farmer’s asking price and market rates.
  - Shows savings compared to market price.
- **Supported Crops:** Rice, Wheat, Tomatoes, Chillies, Sugarcane.
- **Tech Stack:** Python, NumPy, Pandas, Scikit-learn (Random Forest Classifier).

## Installation

Follow these steps to set up AgriAssist locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/prave2-pn/AgriAssist.git
   cd AgriAssist

cd C:\AgriAssist  # Windows
cd /home/user/AgriAssist  # Mac/Linux

pip install numpy pandas scikit-learn

python -c "import numpy, pandas, sklearn; print('All good')"

python agriassist.py

numpy
pandas
scikit-learn
flask

pip install flask

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Simulated Dataset and Model
data = {...}  # Same as your data
df = pd.DataFrame(data)
X = df[['Soil_Moisture', 'Temperature', 'Humidity', 'Pest_Density', 'Pesticide_Usage']]
y = df['Crop_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

crop_data = {...}  # Same as your crop_data
pesticide_data = {...}  # Same as your pesticide_data
valid_crops = ['Rice', 'Wheat', 'Tomatoes', 'Chillies', 'Sugarcane']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/farmer', methods=['GET', 'POST'])
def farmer():
    if request.method == 'POST':
        crop_type = request.form['crop_type']
        pesticide_usage = float(request.form['pesticide_usage'])
        asking_price = float(request.form['asking_price'])
        soil_moisture = np.random.uniform(10, 90)
        temperature = np.random.uniform(20, 40)
        humidity = np.random.uniform(20, 85)
        pest_density = np.random.uniform(0, 90)
        soil_fertility = np.random.uniform(20, 100)
        # Reuse your farmer_crop_health_and_price logic here
        input_data = np.array([[soil_moisture, temperature, humidity, pest_density, pesticide_usage]])
        prediction = model.predict(input_data)[0]
        status = "Good" if prediction == 1 else "Bad"
        # Add rest of your logic (irrigation, pesticide, pricing)
        return render_template('farmer_result.html', crop_type=crop_type, status=status, ...)
    return render_template('farmer.html')

if __name__ == '__main__':
    app.run(debug=True)

    <h1>Welcome to AgriAssist</h1>
<a href="/farmer">Farmer Mode</a>

<h1>Farmer Mode</h1>
<form method="POST">
    Crop Type: <select name="crop_type">
        <option value="Rice">Rice</option>
        <option value="Tomatoes">Tomatoes</option>
        <!-- Add all crops -->
    </select><br>
    Pesticide Usage (L/acre): <input type="number" name="pesticide_usage"><br>
    Asking Price (₹/kg): <input type="number" name="asking_price"><br>
    <input type="submit" value="Scan & Submit">
</form>

python app.py

web: gunicorn app:app

pip install gunicorn
pip freeze > requirements.txt

heroku login
heroku create agriassist-prave2-pn
git init
git add .
git commit -m "Deploy AgriAssist"
git push heroku main

python app.py --host=0.0.0.0



