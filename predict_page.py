from dataclasses import dataclass
import streamlit as st
import numpy as np
import pickle

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data['model']
le_country = data['le_country']
le_education = data['le_education']
le_remote = data['le_remote']

def show_predict_page():
    st.title("Software Dev Salary Predictor")

    st.write("""### We need some information to predict salary""")

    countries = {
        "United States of America",
        "Germany",
        "United Kingdom of Great Britain and Northern Ireland",
        "India",
        "Canada",
        "France",
        "Brazil",
        "Spain",
        "Netherlands",
        "Australia",
        "Italy",
        "Poland",
        "Sweden",
        "Russian Federation",
        "Switzerland"
    }

    education = {
        "Master's degree",
        "Bachelor's degree",
        "Less than a Bachelor's",
        "Post grad"
    }

    remote = {
        "Fully remote",
        "Hybrid (some remote, some in-person)",
        "Full in-person"
    }

    country = st.selectbox("Country", countries)

    ed = st.selectbox("Education", education)

    work = st.selectbox("Type of work", remote)

    exp = st.slider("Years of experience", 0, 50, 3)

    ok = st.button("Calculate Salary")

    if ok:
        X = np.array([[country, ed, work, exp]])
        X[:,0] = le_country.transform(X[:,0])
        X[:,1] = le_education.transform(X[:,1])
        X[:,2] = le_remote.transform(X[:,2])

        X = X.astype(float)

        salary = regressor.predict(X)

        st.subheader(f"The extimated salary is ${salary[0]:.2f}")