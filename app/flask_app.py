from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Define the relative path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'models_pkl', 'RandomForest.pkl')

# Load the pre-trained model
with open(model_path, 'rb') as file:
    model = pickle.load(file)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve inputs from the form
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])

        # Prepare the input for prediction
        input_features = np.array([[temperature, humidity, ph, rainfall, nitrogen, phosphorus, potassium]])

        # Predict the crop
        prediction = model.predict(input_features)
        crop_name = prediction[0]

        return render_template('crop_form.html', crop_name=crop_name)

    return render_template('crop_form.html', crop_name=None)

if __name__ == '__main__':
    app.run(debug=False)
