# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.special import inv_boxcox
from scipy.stats import boxcox_normplot

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Regression",
        page_icon="👋",
    )
    # Display title
    st.title('Regression')

    # Display subheader for raw data
    st.subheader('Raw Data')

    # URL of the CSV file containing insurance data
    csv_url = './data/insurance.csv'
    # Read the CSV data into a DataFrame
    df = pd.read_csv(csv_url)

    # Display the dataset
    st.write(df)

    # Remove duplicate rows from the dataset
    df.drop_duplicates(keep='first', inplace=True)

    # Display numerical plots section
    st.write('### Display Numerical Plots')

    # Select box to choose which numerical feature to plot
    feature_to_plot = st.selectbox('Select a numerical feature to plot', ['age', 'bmi', 'children', 'charges'])

    # Plot the selected numerical feature
    if feature_to_plot:
        st.write(f'Distribution of {feature_to_plot}:')
        fig = plt.figure(figsize=(10, 6))
        plt.hist(df[feature_to_plot], bins=30, color='skyblue', edgecolor='black')
        plt.xlabel(feature_to_plot)
        plt.ylabel('Count')
        st.pyplot(fig)

    # Display categorical plots section
    st.write('### Display Categorical Plots')

    # Select box to choose which categorical feature to plot
    feature_to_plot = st.selectbox('Select a feature to plot', ['sex', 'smoker', 'region'])

    # Plot the selected categorical feature
    if feature_to_plot:
        st.write(f'Distribution of {feature_to_plot}:')
        bar_chart = st.bar_chart(df[feature_to_plot].value_counts())

    # Display relationships section
    st.write('### Display Relationships')

    # Create dropdown menus for user selection of variables
    x_variable = st.selectbox('Select x-axis variable:', df.columns)
    y_variable = st.selectbox('Select y-axis variable:', df.columns)
    color_variable = st.selectbox('Select color variable:', df.columns)
    size_variable = st.selectbox('Select size variable:', df.columns)

    # Create scatter plot using Plotly Express
    fig = px.scatter(df, x=x_variable, y=y_variable, color=color_variable, size=size_variable, hover_data=[color_variable])

    # Display the plot
    st.plotly_chart(fig)

    # Encode categorical variables 'sex', 'smoker', and 'region'
    df['sex_encode'] = LabelEncoder().fit_transform(df['sex'])
    df['smoker_encode'] = LabelEncoder().fit_transform(df['smoker'])
    df['region_encode'] = LabelEncoder().fit_transform(df['region'])

    # Transform the 'charges' variable using Box-Cox transformation
    df['charges_transform'], lambda_value = stats.boxcox(df['charges'])

    # Define features (X) and target (y) and remove duplicate features that will not be used in the model
    X = df.drop(['sex', 'smoker', 'region', 'charges', 'charges_transform'], axis=1)
    y = df['charges_transform']

    # Split the dataset into training and testing sets (90% train, 10% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # Instantiate a linear regression model
    linear_model = LinearRegression()

    # Fit the model using the training data
    linear_model.fit(X_train, y_train)

    # Predict charges for the test set
    y_pred = linear_model.predict(X_test)

    # Create section for user to predict their own charges
    st.write('## Predict Your Own Charges')

    # User input for features
    age = st.slider('Age', min_value=df['age'].min(), max_value=df['age'].max(), value=int(df['age'].mode()))
    bmi = st.slider('BMI', min_value=df['bmi'].min(), max_value=df['bmi'].max(), value=df['bmi'].mean())
    children = st.slider('Number of Children', min_value=df['children'].max(), max_value=df['children'].max(), value=0, format="%d")
    sex = st.selectbox('Sex', ['male', 'female'])
    smoker = st.selectbox('Smoker', ['yes', 'no'])
    region = st.selectbox('Region', ['southwest', 'northwest', 'southeast', 'northeast'])

    # Encode categorical variables for user input
    sex_encode = 1 if sex == 'female' else 0
    smoker_encode = 1 if smoker == 'yes' else 0
    region_encode = ['southwest', 'northwest', 'southeast', 'northeast'].index(region)

    # Predict charges for user input
    predicted_charges_transformed = linear_model.predict([[age, bmi, children, sex_encode, smoker_encode, region_encode]])

    # Reverse the Box-Cox transformation to get the predicted charges
    predicted_charges = inv_boxcox(predicted_charges_transformed, lambda_value)

    # Display the predicted charges
    st.write('Predicted Charges:', round(predicted_charges[0], 0))


    

if __name__ == "__main__":
    run()
