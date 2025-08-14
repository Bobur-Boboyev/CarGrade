import streamlit as st
import pickle
import pandas as pd

with open("models/DecisionTree.pkl", 'rb') as f:
    model = pickle.load(f)

with open("models/LabelEncoders.pkl", 'rb') as f:
    label_encoders = pickle.load(f)

with open("models/TargetEncoder.pkl", 'rb') as f:
    target_encoder = pickle.load(f)

st.set_page_config(page_title="CarGrade", page_icon="🚗", layout="centered")
st.title("🚗 CarGrade - machine Evaluation")
st.write("Evaluate the car using its features")

st.subheader("🔍 Select car features:")

buying = st.selectbox("Buying price", ['vhigh', 'high', 'med', 'low'])
maint = st.selectbox("Price of the maintenance", ["vhigh", 'high', 'med', 'low'])
doors = st.selectbox("Number of doors", ["2", '3', '4', '5more'])
persons = st.selectbox("How many people can fit in a car?", ['2', '4', 'more'])
lug_boot = st.selectbox("the size of luggage boot", ['small', 'med', 'big'])
safety = st.selectbox("security level", ['low', 'med', 'high'])

data = {
    "buying":buying,
    'maint':maint,
    'doors':doors,
    'persons':persons,
    'lug_boot':lug_boot,
    'safety':safety
}
if st.button("Predict"):
    for col, le in label_encoders.items():
        data[col] = le.transform([data[col]])[0]  
    df = pd.DataFrame([data])
    
    pred = model.predict(df)[0]
    pred_target = target_encoder.inverse_transform([pred])[0]
    if pred_target == 'unacc':
        st.warning("🚗 This car is unacceptable.")
    elif pred_target == 'acc':
        st.info("🚗 This car is acceptable.")
    elif pred_target == 'good':
        st.success("🚗 This car is good.")
    else:
        st.success("🚗 This car is very good.")

