import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
import pickle
# Load the trained model
model = pickle.load(open("ln.pkl", "rb"))

# Streamlit UI for user input
st.title("Car Mileage Prediction")

# Collect user input
cyl = st.number_input("Enter the number of cylinders (cyl)", min_value=1, max_value=20, step=1)
disp = st.number_input("Enter the displacement (disp)", min_value=10, max_value=1000, step=10)
hp = st.number_input("Enter the horsepower (hp)", min_value=1, max_value=500, step=1)
drat = st.number_input("Enter the rear axle ratio (drat)", min_value=1.0, max_value=5.0, step=0.1)
wt = st.number_input("Enter the weight (wt)", min_value=1.0, max_value=10.0, step=0.1)
qsec = st.number_input("Enter the 1/4 mile time (qsec)", min_value=10.0, max_value=30.0, step=0.1)
vs = st.number_input("Enter the engine (0 = V-shaped, 1 = straight)", min_value=0, max_value=1, step=1)
am = st.number_input("Enter the transmission (0 = automatic, 1 = manual)", min_value=0, max_value=1, step=1)
gear = st.number_input("Enter the number of forward gears (gear)", min_value=1, max_value=6, step=1)
carb = st.number_input("Enter the number of carburetors (carb)", min_value=1, max_value=8, step=1)

# Preprocess input data
input_data = np.array([[cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb]])

# Predict car mileage
if st.button("Predict Mileage"):
    prediction = model.predict(input_data)
    st.write(f"The predicted mileage of the car is: {prediction[0]:,.2f} mpg") 
