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
year_built = st.sidebar.number_input("Year Built", 1900, 2025, 1990)
waterfront = st.sidebar.selectbox("Waterfront", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
garage_spaces = st.sidebar.number_input("Garage Spaces", min_value=0, max_value=10, value=2, step=1)
#change the below to a no or yes 
basement = st.sidebar.selectbox("Basement", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
central_air = st.sidebar.selectbox("Central Air", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
above_grade_sqft = st.sidebar.number_input("Above Grade SqFt", min_value=500, max_value=10000, value=1500)
below_grade_sqft = st.sidebar.number_input("Below Grade SqFt", min_value=0, max_value=5000, value=500)
mortgage_rate = st.sidebar.number_input("Mortgage Rate", min_value=0.0, max_value=10.0, value=3.5, step=0.1, format="%.1f")

user_input = [[bedrooms, bathrooms, year_built, waterfront, garage_spaces, basement, central_air, above_grade_sqft, below_grade_sqft, mortgage_rate]]

@st.cache_data
def load_data():
    return pd.read_csv("aberdeen_home_sales.csv")

data = load_data()
st.write("Data Preview:", data.head())
st.write("Total records:", len(data))

#convert data types to numeric
data["Bedrooms"] = pd.to_numeric(data["Bedrooms"], errors="coerce")
data["Baths"] = pd.to_numeric(data["Baths"], errors="coerce")
data["YearBuilt"] = pd.to_numeric(data["YearBuilt"], errors="coerce")
data["SoldPrice"] = pd.to_numeric(data["SoldPrice"], errors="coerce")
data["Financing"] = pd.to_numeric(data["Financing"], errors="coerce")

# Ensure Consistency in Column Naming
data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))

# Train model using selected features
features = ["Bedrooms", "Baths", "YearBuilt", "Waterfront", "GarageSpaces", "Basement", "central_air", "AboveGradeSqFt", "BelowGradeSqFt", "MortgageRate"]
target = "SoldPrice"

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

#run streamlit run streamlit_app.py in terminal
