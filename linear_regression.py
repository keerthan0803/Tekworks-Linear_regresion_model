import streamlit as st
import numpy as np
import pandas as pd
#display title
st.title("Linear Regression")

df = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if df is None:
    st.warning("Please upload a dataset to proceed.")
else:
    data = pd.read_csv(df)
    

    # Select target variable
    target_variable = st.selectbox("Select the target variable (dependent variable)", data.columns)

    # select the columns to drop
    columns_to_drop = st.multiselect("Select columns to drop (optional)", data.columns)
    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)
    # Select feature variables
    feature_variables = st.multiselect("Select the feature variables (independent variables)", [col for col in data.columns if col != target_variable])

    if not feature_variables:
            st.warning("Please select at least one feature variable.")
    else:
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split

            X = data[feature_variables]
            y = data[target_variable]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            st.success("Model trained successfully!")

            # Predict the user inputs
            user_input = {}
            for feature in feature_variables:
                user_input[feature] = st.number_input(f"Enter value for {feature}")
            if st.button("Predict"):
                input_data = np.array([user_input[feature] for feature in feature_variables]).reshape(1, -1)
                prediction = model.predict(input_data)
                st.write("Predicted value:", prediction[0])