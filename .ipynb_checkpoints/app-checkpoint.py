import streamlit as st
import pandas as pd
import joblib

# Load your data
df = pd.read_csv('df_model.csv')

# Set the background color and text color
st.markdown(
    """
    <style>
    body {
        color: #788936;
        background-color: #788936;
    }
    .stSlider>div>div>div>div {
        background-color: #788936;
    }
    .st-rc-slider.stSlider>div {
        background-color: #788936;
    }
    .stButton>button {
        background-color: #788936;
        color: white;
    }
    .stSelectbox>div>div>div {
        color: #788936;
        background-color: white;
    }
    .stRadio>div>div>div>label>div:nth-child(2) {
        color: #788936;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define the main function to create and run the app
def main():
    st.title('Project Success Predictor')

    # Add an image
    image = 'What makes International Aid Projects successful (1).png'  # Replace 'your_image_path.jpg' with the path to your image file
    st.image(image, use_column_width=True)

    # Add input fields for each feature

    # Slider for Project Size (USD)
    project_size = st.slider('Project Size (USD)', min_value=0, max_value=int(df['project_size_USD_calculated'].max()), value=int(df['project_size_USD_calculated'].mean()), key='project_size', help='Set the project size in USD')

    # Slider for Start Year
    startyear = st.slider('Start Year', min_value=int(df['startyear'].min()), max_value=int(df['startyear'].max()), value=int(df['startyear'].median()), key='startyear', help='Set the start year')

    # Slider for Evaluation Year
    evalyear = st.slider('Evaluation Year', min_value=int(df['evalyear'].min()), max_value=int(df['evalyear'].max()), value=int(df['evalyear'].median()), key='evalyear', help='Set the evaluation year')

    # Slider for Evaluation Lag (Days)
    eval_lag = st.slider('Evaluation Lag (Days)', min_value=-30, max_value=365*5, value=int(df['eval_lag'].median()), key='eval_lag', help='Set the evaluation lag in days')

    # Slider for Project Duration in Days
    project_duration = st.slider('Project Duration (Days)', min_value=0, max_value=3640, value=int(df['project_duration'].median()), key='project_duration', help='Set the project duration in days')

    donor = st.selectbox('Donor', df['donor'].unique())

    # Dropdown for Country Code
    country_codes = df['country_code_WB'].unique()  
    country_code = st.selectbox('Country Code', options=country_codes)

    # Radio buttons for Region
    region = st.radio('Region', df['region'].unique(), key='region')

    # Radio buttons for Evaluation Type (formerly External Evaluator)
    evaluation_types = df['external_evaluator'].unique()
    evaluation_type = st.radio('Evaluation Type', evaluation_types, key='evaluation_type')

    # Dropdown for Sector (formerly Grouped Category) without NaN values
    sectors = df['Grouped Category'].dropna().unique()
    sector = st.selectbox('Sector', sectors, key='sector')

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
        'external_evaluator': [evaluation_type]
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
