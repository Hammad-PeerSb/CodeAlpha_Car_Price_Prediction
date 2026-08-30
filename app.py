import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AutoValuate | Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LIGHT BLUE & PURPLE UI STYLING ---
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
    }
    
    /* Header Card */
    .header-box {
        background: linear-gradient(135deg, #E0F2FE 0%, #F3E8FF 100%);
        border: 1px solid #C084FC;
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(147, 51, 234, 0.08);
    }
    
    .header-title {
        color: #581C87;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
    }
    
    .header-subtitle {
        color: #0369A1;
        font-size: 1.05rem;
        margin-top: 8px;
        font-weight: 500;
    }
    
    /* Predict Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284C7 0%, #9333EA 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        border: none;
        padding: 14px 28px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(147, 51, 234, 0.25);
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369A1 0%, #7E22CE 100%);
        box-shadow: 0 6px 20px rgba(147, 51, 234, 0.35);
        transform: translateY(-1px);
    }
    
    /* Prediction Output Banner */
    .result-card {
        background: linear-gradient(135deg, #E0F2FE 0%, #F3E8FF 100%);
        border: 2px solid #A855F7;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin: 25px 0;
        box-shadow: 0 8px 24px rgba(168, 85, 247, 0.15);
    }
    
    .result-label {
        color: #0369A1;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin: 0;
    }
    
    .result-value {
        color: #6B21A8;
        font-size: 3rem;
        font-weight: 800;
        margin: 8px 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING WITH CACHING ---
import joblib
import streamlit as st

@st.cache_resource
def load_model():
    try:
        return joblib.load(r"Data\Notebooks\models\car_price_prediction_model.pkl")
    except Exception as e:
        st.error(f"Error loading model.pkl: {e}")
        return None

model = load_model()

# --- HEADER SECTION ---
st.markdown("""
    <div class="header-box">
        <h1 class="header-title">🚘 Auto Price Predictor </h1>
        <p class="header-subtitle">Enterprise-grade car valuation powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# --- VEHICLE INPUT FORM ---
st.markdown("##### 📝 Vehicle Specifications")

col1, col2 = st.columns(2, gap="medium")

with col1:
    # Popular brands list 
    brand_options = ["Toyota", "Honda", "Suzuki", "Hyundai", "Kia", "Nissan", "Ford", "BMW", "Audi"]
    Brand = st.selectbox("🚗 Select Brand / Model", options=sorted(brand_options), index=0)
    
    Year = st.number_input("📅 Manufacturing Year", min_value=1990, max_value=2030, value=2010, step=1)
    
    present_price_pkr = st.number_input(
        "💰 Current Showroom / Present Price (PKR)",
        min_value=100000.0,
        max_value=10000000.0,
        value=350000.0,
        step=50000.0,
        help="Enter current new model market price in PKR."
    )
    
    Driven_kms = st.number_input("🛣️ Total Driven Kilometers", min_value=0, max_value=1000000, value=50000, step=1000)

with col2:
    Fuel_Type = st.selectbox("⛽ Fuel Type", options=["Petrol", "Diesel", "CNG"])
    Selling_type = st.selectbox("🏷️ Seller Type", options=["Dealer", "Individual"])
    Transmission = st.selectbox("⚙️ Transmission Type", options=["Manual", "Automatic"])
    Owner = st.number_input("👤 Number of Previous Owners", min_value=0, max_value=10, value=0, step=1)

# Input Validation
validation_error = None
if present_price_pkr <= 0:
    validation_error = "Present price must be greater than zero."

st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("🚀 Estimate Selling Price", use_container_width=True, key="predict_btn")

# --- INFERENCE & RENDERING ---
if predict_button:
    if validation_error:
        st.error(validation_error)
    elif model is None:
        st.error("Pre-trained model (`model.pkl`) not found. Verify file path.")
    else:
        try:
            # 1. Feature Engineering
            current_year = 2026
            Car_Age = current_year - Year
            Kms_per_Year = Driven_kms / max(Car_Age, 1)

            # 2. Input Feature Scaling (PKR to Lakhs)
            present_price_lakhs = present_price_pkr / 100000.0

            # 3. Construct Feature DataFrame
            input_data = pd.DataFrame([{
                'Brand': Brand,
                'Year': Year,
                'Present_Price': present_price_lakhs,
                'Driven_kms': Driven_kms,
                'Fuel_Type': Fuel_Type,
                'Selling_type': Selling_type,
                'Transmission': Transmission,
                'Owner': Owner,
                'Car_Age': Car_Age,
                'Kms_per_Year': Kms_per_Year
            }])

            # 4. Model Inference & Inverse Scaling (Lakhs to PKR)
            New_prediction = model.predict(input_data)[0]
            predicted_price_pkr = max(0.0, float(New_prediction) * 100000.0)

            # 5. Render Primary Prediction Card
            st.markdown(
                f"""
                <div class="result-card">
                    <p class="result-label"> Estimated Market Value </p>
                    <h1 class="result-value">PKR {predicted_price_pkr:,.2f}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 6. Render Feature Summary Metrics
            st.markdown("##### 📊 Vehicle Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(label="🚗Car Brand", value=Brand)
            m2.metric(label="⌛Car Age", value=f"{Car_Age} Years")
            m3.metric(label="🛣️ Total kilometer", value=f"{Driven_kms:,} km")
            m4.metric(label="⚙️Transmission", value=Transmission)

        except Exception as e:
            st.error(f"Inference Failure: {e}")

# --- FOOTER ---
st.markdown("---")
st.write("PeerSb")
st.caption("AutoPrice • DataScience Project")