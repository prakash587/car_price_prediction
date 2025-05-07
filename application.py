from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('cleaned Car.csv')

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique())
    fuel_types = car['fuel_type'].unique()

    # Mapping company -> model list
    company_model_map = {}
    for company in companies:
        models = sorted(car[car['company'] == company]['name'].unique())
        company_model_map[company] = models

    return render_template('index.html',
                           companies=companies,
                           years=years,
                           fuel_types=fuel_types,
                           company_model_map=company_model_map)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        kms_driven = int(request.form.get('kms_driven'))

        # Create input DataFrame
        input_df = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        prediction = model.predict(input_df)[0]
        return str(int(prediction))
    except Exception as e:
        print("Prediction error:", e)
        return "Error"

if __name__ == '__main__':
    app.run(debug=True)
