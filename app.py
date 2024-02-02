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
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    menu_items=None,
    initial_sidebar_state="collapsed",
)

st.title("Wine Quality Predictor")

# Define paths for images
wine_image = Image.open(normpath("./resources/images/wine-image.jpg"))

# Create two columns for layout
col1, col2 = st.columns([0.45, 0.55], gap="medium")

# Populate col1 with image and project information
with col1:
    st.image(wine_image, use_column_width=True)
    st.write(
        """The application employs web scraping techniques to fetch ebook details
        from **[:blue[eBooks]](https://www.ebooks.com/)** website.
        It then generates a downloadable CSV file for users."""
    )
    st.write(
        """To initiate the process, users select a category,
        a subject, and, if available, a topic. The application then uses
        these selections to scrape the data tailored to the user's preferences."""
    )

# working on col2 section
with col2:
    with st.form("user_inputs"):
        col_3a, col_3b = st.columns([0.5, 0.5], gap="small")
        with col_3a:
            fixed_acidity = st.number_input(
                label="Fixed Acidity:", min_value=3.80, max_value=15.90, value=7.22
            )
            volatile_acidity = st.number_input(
                label="Volatile Acidity:", min_value=0.08, max_value=1.58, value=0.34
            )
            citric_acid = st.number_input(
                label="Citric Acid:", min_value=0.00, max_value=1.66, value=0.32
            )
            residual_sugar = st.number_input(
                label="Residual Sugar:", min_value=0.60, max_value=65.80, value=5.44
            )
            chlorides = st.number_input(
                label="Chlorides:", min_value=0.01, max_value=0.61, value=0.06
            )
            free_sulfur_dioxide = st.number_input(
                label="Free Sulfur Dioxide:",
                min_value=1.00,
                max_value=289.00,
                value=30.53,
            )

        with col_3b:
            total_sulfur_dioxide = st.number_input(
                label="Total Sulfur Dioxide:",
                min_value=6.00,
                max_value=440.00,
                value=115.74,
            )
            density = st.number_input(
                label="Density:", min_value=0.99, max_value=1.04, value=0.99
            )
            pH = st.number_input(
                label="pH:", min_value=2.72, max_value=4.01, value=3.22
            )
            sulphates = st.number_input(
                label="Sulphates:", min_value=0.22, max_value=2.00, value=0.53
            )
            alcohol = st.number_input(
                label="Alcohol:", min_value=8.00, max_value=14.90, value=10.49
            )
            color = st.selectbox("Color:", ("white", "red"))

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
                "color": color,
            }
        ]
        user_df = pd.DataFrame(user_data)
        model_pred = ModelPrediction()
        wine_quality_score = round(model_pred.predict(user_df)[0])
        st.write(
            f"With the given input features, the wine quality score predicted"
            f" by the model is: **:green[{wine_quality_score:0.0f}]**"
        )
