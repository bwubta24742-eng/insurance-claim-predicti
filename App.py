import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = {
    'age': [22,25,30,35,40,45,50,55,60,65],
    'bmi': [21,24,27,29,25,31,28,30,26,32],
    'children': [0,1,2,1,3,2,0,3,1,2],
    'smoker': [0,0,1,0,0,1,0,1,0,1],
    'region': [0,1,2,3,0,1,2,3,0,1],
    'claim': [0,0,1,0,1,1,0,1,0,1]
}

df = pd.DataFrame(data)

X = df.drop('claim', axis=1)
y = df['claim']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.set_page_config(page_title="Insurance Prediction")

st.title("🛡️ Insurance Claim Prediction")
st.write("Enter Your Details")

age = st.number_input("Age", 18, 100, 28)
bmi = st.number_input("BMI", 10.0, 60.0, 26.0)
children = st.number_input("Children", 0, 10, 1)

smoker = st.selectbox("Smoker?", ["No", "Yes"])

region = st.selectbox(
    "Region",
    ["Region 0", "Region 1", "Region 2", "Region 3"]
)

if st.button("Predict Claim"):
    person = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker': [1 if smoker == "Yes" else 0],
        'region': [int(region[-1])]
    })

    result = model.predict(person)

    if result[0] == 1:
        st.warning("Insurance Claim Predicted")
    else:
        st.success("No Insurance Claim Predicted")

st.caption("Educational demo only. Not for real insurance decisions.")
