import streamlit as st
import pandas as pd
import joblib

# Load your data
df = pd.read_csv('df_model.csv')

# Define the main function to create and run the app
def main():
    st.title('Project Success Success Predictor')

    # Add input fields for each feature
    project_size = st.number_input('Project Size (USD)', value=df['project_size_USD_calculated'].mean())
    startyear = st.number_input('Start Year', value=df['startyear'].median())
    evalyear = st.number_input('Evaluation Year', value=df['evalyear'].median())
    eval_lag = st.number_input('Evaluation Lag', value=df['eval_lag'].median())
    project_duration = st.number_input('Project Duration', value=df['project_duration'].median())
    completion_year = st.number_input('Completion Year', value=df['completion_year'].median())

    donor = st.selectbox('Donor', df['donor'].unique())
    country_code = st.selectbox('Country Code', df['country_code_WB'].unique())
    region = st.selectbox('Region', df['region'].unique())
    external_evaluator = st.selectbox('External Evaluator', df['external_evaluator'].unique())
    grouped_category = st.selectbox('Grouped Category', df['Grouped Category'].unique())

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
        'grouped_category': [grouped_category],
        'external_evaluator': [external_evaluator]
    })

    # Drop NaN values from the input data
    df_combined = pd.concat([df, new_data], ignore_index=True)
    df_combined.dropna(inplace=True)

    # Encode the combined data
    df_encoded = pd.get_dummies(df_combined, drop_first=True)

    # Separate the new encoded data
    new_data_encoded = df_encoded.tail(1)

    # Drop columns that are not in model_columns
    new_data_encoded = new_data_encoded[model_columns]# Combine the new data with the original df5 to ensure all columns are present
    df_combined = pd.concat([df, new_data], ignore_index=True)

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