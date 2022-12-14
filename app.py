import streamlit as st
import pandas as pd

from run_training import ModelTraining
from utils.data_processing import inference_input_processing

# Load last trained NN model
MTL_model = ModelTraining()
MTL_model.load_model(load_last_trained_model=True)

# Title
st.header("Inference app for Multi-task-learning")

# Input bar x1
x1 = st.number_input("Enter x1", value=5.62)
# Input bar x2
x2 = st.number_input("Enter x2", value=1.26)
# Input bar x3
x3 = st.number_input("Enter x3", value=-0.49)
# Input bar X4
x4 = st.number_input("Enter x4", value=6.63)
# Input bar x5
x5 = st.number_input("Enter x5", value=3.77)
# Input bar x6
x6 = st.number_input("Enter x6", value=-0.13)
# Input bar z
z = st.number_input("Enter z", value=-0.91)


# If button is pressed
if st.button("Predict"):
    # Store inputs into dataframe
    raw_input = pd.DataFrame([[x1, x2, x3, x4, x5, x6, z]],
                     columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'z'])
    X = inference_input_processing(raw_input)
    # Get prediction
    prediction = MTL_model.inference(X)

    # Output prediction
    st.text(f"Prediction results: \n{prediction}")
