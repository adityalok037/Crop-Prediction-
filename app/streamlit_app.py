import pickle
import numpy as np
import os
import streamlit as st

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'models_pkl', 'RandomForest.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Title of the web app
st.title("Crop Prediction App")
st.subheader("Fill your details please:-")



# Input fields with min and max values, but no default value (empty by default)
temperature = st.number_input('Temperature (°C)',  max_value=60.0, value=None, step=0.1)
humidity = st.number_input('Humidity (%)', value=None, step=0.1)
ph = st.number_input('pH level', min_value=0.0, max_value=14.0, value=None, step=0.1)
rainfall = st.number_input('Rainfall (mm)', min_value=0.0, value=None, step=0.1)
nitrogen = st.number_input('Nitrogen (N)', min_value=0.0, value=None, step=0.1)
phosphorus = st.number_input('Phosphorus (P)', min_value=0.0, value=None, step=0.1)
potassium = st.number_input('Potassium (K)', min_value=0.0, value=None, step=0.1)

# Ensure all inputs are filled before allowing the prediction
if st.button('Predict Crop') and all(value is not None for value in [temperature, humidity, ph, rainfall, nitrogen, phosphorus, potassium]):
    # Prepare the input for prediction
    input_features = np.array([[temperature, humidity, ph, rainfall, nitrogen, phosphorus, potassium]])

    # Predict the crop
    prediction = model.predict(input_features)
    crop_name = prediction[0]

    # Display the result
    st.success(f'The predicted crop is: {crop_name}')
else:
    st.warning('Please fill all inputs before predicting.')
