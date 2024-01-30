#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 22:27:38 2023

@author: chinmayakumarpalo
"""

import joblib
import pandas as pd

# Load the saved scikit-learn model
loaded_model = joblib.load('logistic_model.pkl')

# Create a dictionary to store user input for attribute values
user_input = {}

# List of attributes for prediction
attributes = ['PM2.5', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3', 'Benzene', 'AQI']

# Collect user input for attribute values
for attribute in attributes:
    user_input[attribute] = float(input(f"Enter value for {attribute}: "))

# Create a DataFrame from the user input
new_data = pd.DataFrame([user_input])

# Use the loaded model to make predictions on the new data
predicted_class = loaded_model.predict(new_data)

# Print the predicted class
print("Predicted Class:", predicted_class[0])