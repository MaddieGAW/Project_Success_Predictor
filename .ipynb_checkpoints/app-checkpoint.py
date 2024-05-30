import streamlit as st
import pandas as pd
import joblib

# Load your data
df = pd.read_csv('df_model.csv')

# Define the main function to create and run the app
def main():
    st.title('Project Success Success Predictor')

    # Add input fields for each feature

    # Slider for Project Size (USD)
    project_size = st.slider('Project Size (USD)', min_value=0, max_value=int(df['project_size_USD_calculated'].max()), value=int(df['project_size_USD_calculated'].mean()))

    # Slider for Start Year
    startyear = st.slider('Start Year', min_value=int(df['startyear'].min()), max_value=int(df['startyear'].max()), value=int(df['startyear'].median()))

    # Slider for Evaluation Year
    evalyear = st.slider('Evaluation Year', min_value=int(df['evalyear'].min()), max_value=int(df['evalyear'].max()), value=int(df['evalyear'].median()))

    # Slider for Evaluation Lag (Days)
    eval_lag = st.slider('Evaluation Lag (Days)', min_value=-30, max_value=365*5, value=int(df['eval_lag'].median()))

    # Slider for Project Duration in Days
    project_duration = st.slider('Project Duration (Days)', min_value=0, max_value=3640, value=int(df['project_duration'].median()))

    donor = st.selectbox('Donor', df['donor'].unique())

    # Dropdown for Country Code
    country_codes = df['country_code_WB'].unique()  
    country_code = st.selectbox('Country Code', options=country_codes)

    region = st.selectbox('Region', df['region'].unique())

    # Dropdown for External Evaluator with capitalized options
    external_evaluator = st.selectbox('External Evaluator', options=df['external_evaluator'].unique(), format_func=lambda x: x.capitalize())

    # Dropdown for Sector (formerly Grouped Category) without NaN values
    sectors = df['Grouped Category'].dropna().unique()
    sector = st.selectbox('Sector', sectors)

    # Load the trained model and feature columns
    model_path = 'model.joblib'
    columns_path = 'model_columns.joblib'
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)

    # Prepare new data for prediction
    new_data = pd.DataFrame({
        'project_size_USD_calculated': [project_size],
        'startyear': [startyear],
        'evalyear': [evalyear],
        'eval_lag': [eval_lag],
        'project_duration': [project_duration],
        'donor': [donor],
        'country_code_WB': [country_code],
        'region': [region],
        'Grouped Category': [sector],
        'external_evaluator': [external_evaluator]
    })
    
    # Handle categorical variables
    categorical_columns = ['donor', 'country_code_WB', 'region', 'external_evaluator', 'Grouped Category']
    new_data_encoded = pd.get_dummies(new_data, columns=categorical_columns, drop_first=True)

    # Align the columns of new_data_encoded with model_columns
    new_data_encoded = new_data_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict the success of the project
    if st.button('Predict'):
        prediction = model.predict(new_data_encoded)
        st.subheader('Prediction')
        if prediction[0] == 1:
            st.write('The project is predicted to be successful.')
        else:
            st.write('The project is predicted to not be successful.')

# Run the app
if __name__ == "__main__":
    main()