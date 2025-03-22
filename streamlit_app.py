import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from category_encoders import TargetEncoder

st.set_page_config(page_title="Aberdeen Home Price Predictor", layout="centered")

st.title("🏡 Aberdeen Home Price Predictor")
st.markdown("Enter property details to estimate the home sale price in Aberdeen, MD.")

st.sidebar.header("Property Details")

bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
sqft = st.sidebar.number_input("Square Feet", 500, 10000, 1500)
year_built = st.sidebar.number_input("Year Built", 1900, 2025, 1990)
waterfront = st.sidebar.selectbox("Waterfront", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

user_input = [[bedrooms, bathrooms, sqft, year_built,waterfront]]

@st.cache_data
def load_data():
    return pd.read_csv("aberdeen_home_sales.csv")

data = load_data()
st.write("Data Preview:", data.head())
st.write("Total records:", len(data))

#convert data types to numeric
data["Bedrooms"] = pd.to_numeric(data["Bedrooms"], errors="coerce")
data["Baths"] = pd.to_numeric(data["Baths"], errors="coerce")
data["SqFt"] = pd.to_numeric(data["SqFt"], errors="coerce")
data["YearBuilt"] = pd.to_numeric(data["YearBuilt"], errors="coerce")
data["SoldPrice"] = pd.to_numeric(data["SoldPrice"], errors="coerce")
data["Financing"] = pd.to_numeric(data["Financing"], errors="coerce")

# Ensure Consistency in Column Naming
data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))

# Train model using selected features
features = ["Bedrooms", "Baths", "SqFt", "YearBuilt", "Waterfront"]
target = "SoldPrice"

data[features] = data[features].apply(pd.to_numeric, errors="coerce")  # Convert to numeric, coercing errors

for col in features:
    data[col] = pd.to_numeric(data[col], errors="coerce")  # Convert to numeric, coercing errors

# st.write("Missing values per column:", data.isnull().sum())

# Rename columns for consistency if needed
data = data.rename(columns=lambda x: x.strip())

X = data[features]
y = data[target]

#try gradient boosting regressor since nan values
model = GradientBoostingRegressor()
model.fit(X, y)

if st.sidebar.button("Predict Price"):
    prediction = model.predict(user_input)[0]
    st.success(f"💰 Estimated Sale Price: ${prediction:,.0f}")

with st.expander("📊 Data Insights"):
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data["SoldPrice"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Importance")
    imp_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
    st.bar_chart(imp_df.set_index("Feature"))
