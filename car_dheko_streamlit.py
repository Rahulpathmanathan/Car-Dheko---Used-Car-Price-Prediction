import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


# Load the trained model
model_path = r"D:\Car_Dheko\env\Scripts\Final_model.pkl"
with open(model_path, 'rb') as file:
    Final_model = pickle.load(file)


# Load the label encoders
label_encoder_path = r"D:\Car_Dheko\env\Scripts\Label_encoder.pkl"
with open(label_encoder_path, 'rb') as file:
    encoders = pickle.load(file)

# Load the OneHotEncoder
one_encoder_path = r"D:\Car_Dheko\env\Scripts\Onehot_encoder.pkl"
with open(one_encoder_path, 'rb') as file:
    oe = pickle.load(file)

# Load the MinMaxScaler
minmaxscaler_path = r"D:\Car_Dheko\env\Scripts\minmaxscaler.pkl"
with open(minmaxscaler_path, 'rb') as file:
    scaler = pickle.load(file)


st.set_page_config(page_title='Used Car Price Prediction',layout="centered")

st.markdown("<div style='text-align: center;'><h1>Used Car Price Prediction</h1></div>", unsafe_allow_html=True)

df = pd.read_csv(r"D:\Car_Dheko\env\Scripts\final.csv")

col1, col2, col3 = st.columns(3)

with col1:
    Body_type=st.selectbox("Select a Body Type:",df["Body_Type"].unique())
    city=st.selectbox("Select a City:",df["City"].unique())
    Fuel_type = st.selectbox("Select a Fuel Type:",df["Fuel_Type"].unique())

with col2:
    Brand = st.selectbox("Select a Brand:",df["Brand"].unique())
    filtered_models = df[df["Brand"] == Brand]["Model"].unique()
    Model = st.selectbox("Select a Model:", filtered_models)
    Year = st.selectbox("Model_Year:",df["Model_Year"].unique())

with col3:

    owners = st.number_input("owners", min_value=1, max_value=10)
    Seats=st.selectbox("Seats",df["Seats"].unique())
    Kilometers = st.slider("kilometers:",min_value=0, max_value=100000)
    


Encoded_fuel_type = encoders.get('Fuel_Type').transform([Fuel_type])
Encoded_Body_type = encoders.get('Body_Type').transform([Body_type])
Encoded_city = encoders.get('City').transform([city])
Encoded_kilometers = scaler.transform([[Kilometers]])
brand_and_model = [(Brand,Model)]
Encoded_brand_model = oe.transform(brand_and_model)
Encoded_brand_model_list = Encoded_brand_model.flatten().tolist()

input_features = [
    Encoded_Body_type[0],       
    Encoded_kilometers[0][0],   
    owners,
    Year,
    Encoded_fuel_type[0],       
    Seats,
    Encoded_city[0]             
] + Encoded_brand_model_list


input_features = np.array(input_features).reshape(1, -1)

if st.button("predict"):
    prediction = Final_model.predict(input_features)
    st.success(f"The predicted price of the car is ₹ {prediction[0]:,.2f} Lakhs")

