import streamlit as st
import pandas as pd
import joblib as jb

st.title('Simple ML app for house price')

model_dt = jb.load("models/iowa_model.pkl")
model_rf = jb.load("models/iowa_model_rf.pkl")
features = jb.load("models/iowa_features.pkl")

#['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
lot_area = st.number_input('Lot Area', value=8000)
year_built = st.number_input('Year Built', value=2000)
first_floor_sf = st.number_input('1st Floor SF', value=800)
second_floor_sf = st.number_input('2nd Floor SF', value=800)
full_bath = st.number_input('Full Bath', value=2)
bedroom_abv_gr = st.number_input('Bedroom Abv Gr', value=3)
total_rooms_abv_grd = st.number_input('Total Rooms Abv Grd', value=6)

option = st.selectbox('Select Model', ['Decision Tree', 'Random Forest'])

if option == 'Decision Tree' and st.button('Predict Price'):
    #transform this to df for predictions
    input_data_list = [lot_area, year_built, first_floor_sf, second_floor_sf, full_bath, bedroom_abv_gr, total_rooms_abv_grd]
    input_df = pd.DataFrame([input_data_list], columns=features)
    prediction_dt = model_dt.predict(input_df)[0]
    st.write(f"Decision Tree Predicted Price: $**{prediction_dt:,.2f}**")

if option == 'Random Forest' and st.button('Predict Price'):
    #transform this to df for predictions
    input_data_list = [lot_area, year_built, first_floor_sf, second_floor_sf, full_bath, bedroom_abv_gr, total_rooms_abv_grd]
    input_df = pd.DataFrame([input_data_list], columns=features)
    prediction_rf = model_rf.predict(input_df)[0]
    st.write(f"Random Forest Predicted Price: $**{prediction_rf:,.2f}**")