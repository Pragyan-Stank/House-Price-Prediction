import streamlit as st
import pandas as pd
import numpy as np
import pickle  # Using pickle instead of joblib

# Load the saved model and scaler
# Make sure you save your model and scaler to files after training
try:
    with open('best_xgb_model.pkl', 'rb') as f:
        best_xgb_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('numerical_cols.pkl', 'rb') as f:
        numerical_cols = pickle.load(f)
    with open('ocean_proximity_cols.pkl', 'rb') as f:
        ocean_proximity_cols = pickle.load(f)
    with open('training_columns.pkl', 'rb') as f:
        training_columns = pickle.load(f)


except FileNotFoundError:
    st.error("Model or scaler files not found. Please train the model and save them first.")
    st.stop() # Stop execution if files are missing

# Define the Streamlit app
st.title("House Price Prediction App")

st.write("Enter the features of the house to predict its price.")

# Create input fields for the features
longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=41.0)
total_rooms = st.number_input("Total Rooms", value=880.0)
total_bedrooms = st.number_input("Total Bedrooms", value=129.0)
population = st.number_input("Population", value=322.0)
households = st.number_input("Households", value=126.0)
median_income = st.number_input("Median Income", value=8.3252)
ocean_proximity = st.selectbox("Ocean Proximity", ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'])

# Create a dictionary from the input values
input_data = {
    'longitude': longitude,
    'latitude': latitude,
    'housing_median_age': housing_median_age,
    'total_rooms': total_rooms,
    'total_bedrooms': total_bedrooms,
    'population': population,
    'households': households,
    'median_income': median_income,
    'ocean_proximity': ocean_proximity, # Make sure ocean_proximity is in the dictionary
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# --- Preprocessing the input data ---

# Engineer features (same as in your notebook)
input_df['rooms_per_household'] = input_df['total_rooms'] / input_df['households']
input_df['bedrooms_per_room'] = input_df['total_bedrooms'] / input_df['total_rooms']
input_df['population_per_household'] = input_df['population'] / input_df['households']
# Apply log transformation to median_income
input_df['median_income_log'] = np.log1p(input_df['median_income'])
input_df['income_rooms_interaction'] = input_df['median_income'] * input_df['rooms_per_household']


# One-hot encode 'ocean_proximity'
# Create dummy columns for all possible ocean_proximity values
# Ensure the 'ocean_proximity' column exists in input_df at this point
ocean_proximity_dummies = pd.get_dummies(input_df['ocean_proximity'], prefix='ocean_proximity')

# Ensure all one-hot encoded columns are present, filling missing ones with 0
for col in ocean_proximity_cols:
    if col not in ocean_proximity_dummies.columns:
        ocean_proximity_dummies[col] = 0

# Reorder the one-hot encoded columns to match the training data
ocean_proximity_dummies = ocean_proximity_dummies[ocean_proximity_cols]

# Drop the original 'ocean_proximity' column
input_df = input_df.drop('ocean_proximity', axis=1)

# Concatenate the one-hot encoded columns
input_df = pd.concat([input_df, ocean_proximity_dummies], axis=1)

# Select the numerical columns for scaling (same as in your notebook)
# Make sure the order of columns is the same as during training
input_numerical_cols = input_df.select_dtypes(include=['number']).columns.tolist()
# Remove one-hot encoded columns from the list
input_numerical_cols = [col for col in input_numerical_cols if 'ocean_proximity' not in col]


# Scale the numerical features
# Make sure the numerical_cols list is correct and only contains columns to be scaled
input_df[input_numerical_cols] = scaler.transform(input_df[input_numerical_cols])

# Ensure the order of columns in input_df matches the order of columns in X_train
# This is crucial for the model prediction
# Use the loaded training_columns to reindex the input_df
input_df = input_df.reindex(columns=training_columns, fill_value=0)


# --- Make prediction ---
if st.button("Predict Price"):
    # Predict the log-transformed price
    predicted_price_log = best_xgb_model.predict(input_df)[0]

    # Inverse transform the log-transformed price
    predicted_price = np.expm1(predicted_price_log)

    st.success(f"Predicted House Price: ${predicted_price:,.2f}")