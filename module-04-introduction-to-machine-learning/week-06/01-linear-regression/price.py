import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Load Saved Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open("price_of_house.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: 'price_of_house.pkl' file not found. Please ensure it is in the same directory.")
        return None

model = load_model()

# ---------------------------------------------------------
# Custom CSS for UI Enhancements
# ---------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 20px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar - Inputs
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/illustrations/100/house.png", width=100)
st.sidebar.title("🏡 Property Details")
st.sidebar.write("Adjust the features below to estimate the property price.")

# Feature Inputs
bedrooms = st.sidebar.slider("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.sidebar.slider("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
area_sqm = st.sidebar.number_input("Area (in Sq Meters)", min_value=20, max_value=1000, value=120, step=5)

location_tier = st.sidebar.selectbox(
    "Location Tier",
    options=[1, 2, 3],
    format_func=lambda x: f"Tier {x} ({'Standard' if x==1 else 'Premium' if x==2 else 'Luxury'})"
)

age_years = st.sidebar.slider("Property Age (Years)", min_value=0, max_value=100, value=10, step=1)

predict_btn = st.sidebar.button("✨ Predict Price")

# ---------------------------------------------------------
# Main Page Body
# ---------------------------------------------------------
st.title("🏠 Real Estate Price Estimation")
st.markdown("Estimate house values instantly based on key property characteristics.")

st.divider()

# Arrange content into columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Selected Specifications")
    
    # Create a nice summary DataFrame to display selected inputs
    input_summary = pd.DataFrame({
        "Feature": ["Bedrooms", "Bathrooms", "Area (sqm)", "Location Tier", "Age (years)"],
        "Value": [bedrooms, bathrooms, area_sqm, f"Tier {location_tier}", f"{age_years} yrs"]
    })
    
    st.dataframe(input_summary, hide_index=True, use_container_width=True)

with col2:
    st.subheader("💰 Estimated Valuation")
    
    if predict_btn or 'predicted' in st.session_state:
        st.session_state['predicted'] = True
        
        if model is not None:
            # Prepare feature array in exact order: [bedrooms, bathrooms, area_sqm, location_tier, age_years]
            features = np.array([[bedrooms, bathrooms, area_sqm, location_tier, age_years]])
            
            # Predict
            prediction = model.predict(features)[0]
            
            # Format outputs (assumes outcome is positive)
            predicted_price = max(0, float(prediction))
            
            # Display Prediction Card
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #6c757d; margin-bottom: 8px;">Estimated Market Value</h3>
                    <h1 style="color: #2E7D32; font-size: 2.5rem; margin: 0;">₦{predicted_price:,.2f} Million</h1>
                    <p style="color: #8d99ae; font-size: 0.9rem; margin-top: 8px;">≈ ₦{predicted_price * 1_000_000:,.0f}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.success("Prediction calculated successfully!")
            st.info("💡 **Note:** Real estate estimates can vary depending on market dynamics and micro-locations.")
    else:
        st.info("👈 Set property details in the sidebar and click **'Predict Price'** to see the result.")

# ---------------------------------------------------------
# Footer / Dataset Sample Section
# ---------------------------------------------------------
st.divider()
with st.expander("📊 View Sample Dataset Context"):
    st.write("This model was trained on dataset features structured like this:")
    sample_df = pd.DataFrame({
        'bedrooms': [3, 3, 5, 4, 2],
        'bathrooms': [2, 3, 5, 4, 2],
        'area_sqm': [88, 152, 204, 146, 54],
        'location_tier': [1, 1, 2, 1, 1],
        'age_years': [25, 25, 23, 1, 18],
        'price_million_naira': [108.0, 122.7, 205.8, 136.9, 67.9]
    })
    st.dataframe(sample_df, hide_index=True, use_container_width=True)