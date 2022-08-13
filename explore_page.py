import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = "Other"
            
    return categorical_map

def cleanExp(x):
    if x == "Less than 1 year":
        return 0.5
    elif x == "More than 50 years":
        return 50
    return float(x)

def cleanEd(x):
    if "Bachelor" in x:
        return "Bachelor's degree"
    if "Master" in x:
        return "Master's degree"
    if "Professional" in x or "Other doctoral" in x:
        return "Post grad"
    return "Less than a Bachelor's"

@st.cache
def load_data():
    df = pd.read_csv('survey_results_public.csv')

    df = df[['Country', 'EdLevel', 'YearsCodePro', 'Employment', 'RemoteWork', 'ConvertedCompYearly']]
    df = df.rename({'ConvertedCompYearly':'Salary'}, axis=1)

    df = df[df['Salary'].notnull()]

    df = df.dropna()

    df = df[df['Employment'] == 'Employed, full-time']
    df = df.drop('Employment', axis=1)

    country_map = shorten_categories(df.Country.value_counts(), 400)
    df['Country'] = df['Country'].map(country_map)

    df = df[df['Salary'] <= 300000]
    df = df[df['Salary'] >= 10000]
    df = df[df['Country'] != 'Other']

    df['YearsCodePro'] = df['YearsCodePro'].apply(cleanExp)

    df['EdLevel'] = df['EdLevel'].apply(cleanEd)

    return df

df = load_data()

def show_explore_page():
    st.title("Explore Software dev's salaries!")

    st.write("""\n### Stack Overflow Developer Survey 2022\n""")

    data = df['Country'].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")

    st.write("""\n#### No of data from different countries""")

    st.pyplot(fig1)

    st.write("""\n#### Mean salary based on Country""")

    data = df.groupby(['Country'])['Salary'].mean().sort_values(ascending=True)

    st.bar_chart(data)

    st.write("""\n#### Mean salary based on Experience""")

    data = df.groupby(['YearsCodePro'])['Salary'].mean().sort_values(ascending=True)

    st.line_chart(data)