import streamlit as st
import pandas as pd
import joblib as jb

st.title('Simple ML app for house price')

model = jb.load("iowa_model.pkl")
features = jb.load("iowa_features.pkl")



