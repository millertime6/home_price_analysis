import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Aberdeen Home Price Predictor", layout="centered")

st.title("🏡 Aberdeen Home Price Predictor")
st.markdown("Enter property details to estimate the home sale price in Aberdeen, MD.")

st.sidebar.header("Property Details")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
sqft = st.sidebar.number_input("Square Footage", 500, 10000, 1500)
year_built = st.sidebar.number_input("Year Built", 1900, 2025, 1990)

user_input = [[bedrooms, bathrooms, sqft, year_built]]

if st.sidebar.button("Predict Price"):
    prediction = model.predict(user_input)[0]
    st.success(f"💰 Estimated Sale Price: ${prediction:,.0f}")

@st.cache_data
def load_data():
    return pd.read_csv("data/aberdeen_home_sales.csv")

data = load_data()

with st.expander("📊 Data Insights"):
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data["price"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Importance")
    importances = model.coef_
    features = ["Bedrooms", "Bathrooms", "SqFt", "Year Built"]
    imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
    st.bar_chart(imp_df.set_index("Feature"))

