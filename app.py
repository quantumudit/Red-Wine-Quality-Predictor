"""_summary_
"""

# Import necessary libraries
from os.path import normpath

import pandas as pd
import streamlit as st
from PIL import Image

from src.components.model_prediction import ModelPrediction

# Configure the Streamlit page
st.set_page_config(
    page_title="Red Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    menu_items=None,
    initial_sidebar_state="collapsed",
)

st.title("Red Wine Quality Predictor")

# Define paths for images
wine_image = Image.open(normpath("./resources/images/wine-image.jpg"))

# Create two columns for layout
col1, col2 = st.columns([0.45, 0.55], gap="medium")

# Populate col1 with image and project information
with col1:
    st.image(wine_image, use_column_width=True)
    st.write(
        """The application utilizes a machine learning model
        to assess the quality score of red wine based on a range of input features
        """
    )
    st.write(
        """To obtain the desired outcome, input the appropriate values into
        the designated field.
        """
    )

# working on col2 section
with col2:
    with st.form("user_inputs"):
        col_3a, col_3b = st.columns([0.5, 0.5], gap="small")
        with col_3a:
            fixed_acidity = st.number_input(
                label="Fixed Acidity:", min_value=1.00, max_value=25.00, value=7.22
            )
            volatile_acidity = st.number_input(
                label="Volatile Acidity:", min_value=0.01, max_value=5.00, value=0.34
            )
            citric_acid = st.number_input(
                label="Citric Acid:", min_value=0.00, max_value=3.00, value=0.32
            )
            residual_sugar = st.number_input(
                label="Residual Sugar:", min_value=0.10, max_value=99.99, value=5.44
            )
            chlorides = st.number_input(
                label="Chlorides:", min_value=0.01, max_value=2.00, value=0.06
            )
            free_sulfur_dioxide = st.number_input(
                label="Free Sulfur Dioxide:",
                min_value=1.00,
                max_value=400.00,
                value=30.53,
            )

        with col_3b:
            total_sulfur_dioxide = st.number_input(
                label="Total Sulfur Dioxide:",
                min_value=1.00,
                max_value=600.00,
                value=115.74,
            )
            density = st.number_input(
                label="Density:", min_value=0.01, max_value=3.00, value=0.99
            )
            pH = st.number_input(
                label="pH:", min_value=0.01, max_value=9.99, value=3.22
            )
            sulphates = st.number_input(
                label="Sulphates:", min_value=0.00, max_value=9.00, value=0.53
            )
            alcohol = st.number_input(
                label="Alcohol:", min_value=5.00, max_value=25.00, value=10.49
            )

        st.form_submit_button()
        user_data = [
            {
                "fixed_acidity": fixed_acidity,
                "volatile_acidity": volatile_acidity,
                "citric_acid": citric_acid,
                "residual_sugar": residual_sugar,
                "chlorides": chlorides,
                "free_sulfur_dioxide": free_sulfur_dioxide,
                "total_sulfur_dioxide": total_sulfur_dioxide,
                "density": density,
                "pH": pH,
                "sulphates": sulphates,
                "alcohol": alcohol,
            }
        ]
        user_df = pd.DataFrame(user_data)
        model_pred = ModelPrediction()
        wine_quality_score = round(model_pred.predict(user_df)[0])
        st.write(
            f"With the given input features, the wine quality score predicted"
            f" by the model is: **:green[{wine_quality_score:0.0f}]**"
        )
