import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from Target_encoder import TargetEncoder


# ---------------------------------------------------------
#  CACHE: Load files only ONCE (prevents freezing)
# ---------------------------------------------------------
@st.cache_data
def load_names():
    return pd.read_csv("Car_State_name.csv")

@st.cache_resource
def load_encoder():
    with open("target_encoder.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model():
    try:
        with open("CHEVROLET_DAEWOO_RAVON_LGBM_logtuned.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠ Model file not found!")
        return None


# ---------------------------------------------------------
#  LOAD DATA
# ---------------------------------------------------------

all_names = load_names()
encoder = load_encoder()
model = load_model()


# ---------------------------------------------------------
#  SAFE TIME INDEX FUNCTION
# ---------------------------------------------------------
# def time_index(df):

#     if "month_year" not in df.columns:
#         raise ValueError("'month_year' is missing!")

#     df["month_year"] = pd.to_datetime(df["month_year"])

#     # For 1-row input: no need sorting
#     df["time_index"] = (
#         (df["month_year"].dt.year - df["month_year"].dt.year.min()) * 12 +
#         df["month_year"].dt.month
#     )

#     df["month"] = df["month_year"].dt.month
#     df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
#     df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

#     return df.drop(columns=["month_year", "month"])

# def add_time_features_from_month_year(df: pd.DataFrame, col: str) -> pd.DataFrame:
#     # Convert month_year to datetime
#     df["date"] = pd.to_datetime(df[col], format="%Y-%m", errors="coerce")
    
#     # Extract year and month
#     df["year"] = df["date"].dt.year
#     df["month"] = df["date"].dt.month
    
#     # Sort by date
#     df = df.sort_values("date").reset_index(drop=True)
    
#     # Time index: counts months since first entry
#     df["time_idx"] = (df["year"] - df["year"].min()) * 12 + (df["month"] - df["month"].min())
    
#     # Cyclical encoding for month
#     df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
#     df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
#     return df.drop(columns=['month','month_year','date'])

# # ---------------------------------------------------------
#  STREAMLIT PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="auto_model",
    # page_icon="",
    layout="wide"
)

st.title("car_model")
st.markdown("---")


# ---------------------------------------------------------
#  DROPDOWN LISTS
# ---------------------------------------------------------

CAR_NAMES = all_names["car_name"].unique().tolist()
REGIONS = all_names["state"].unique().tolist()
FUEL_TYPES = all_names["fuel_type"].unique().tolist()
ITEM_TYPES = all_names["item_type"].unique().tolist()
BODY_TYPES = all_names["body_type"].unique().tolist()
COLORS = all_names["color"].unique().tolist()
CAR_CONDITIONS = all_names["car_condition"].unique().tolist()
OWNERS_COUNT = ["1", "2", "3", "4"]


# ---------------------------------------------------------
#  INPUT SECTION
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    # st.subheader("info")

    car_name = st.selectbox("Car Name", CAR_NAMES)
    state = st.selectbox("Region/State", REGIONS)

    release_year = st.number_input(
        "Release Year",
        min_value=1990,
        max_value=datetime.now().year,
        value=2015,
        step=1
    )
    # current_yaer = st.number_input(
    #     'current year',
    #     min_value=2010,
    #     max_value=2025,
    #     value=2025,
    #     step=1        

    # )

    transmission = st.radio(
        "Transmission",
        options=[0, 1],
        format_func=lambda x: "Automatic" if x == 0 else "Manual",
        horizontal=True
    )

    mileage = st.number_input(
        "Mileage (km)",
        min_value=0.0,
        max_value=500000.0,
        value=50000.0,
        step=1000.0
    )

    

    engine_volume = st.number_input(
        "Engine Volume (L)",
        min_value=0.5,
        max_value=8.0,
        value=1.5,
        step=0.1
    )

    brand_type = st.selectbox(
        "Brand Type",
        options=[1,0],
        
    )

with col2:
    # st.subheader("info")

    fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
    item_type = st.selectbox("Item Type", ITEM_TYPES)
    body_type = st.selectbox("Body Type", BODY_TYPES)
    owners_count = st.selectbox("Number of Previous Owners", OWNERS_COUNT)
    car_condition = st.selectbox("Car Condition", CAR_CONDITIONS)
    color = st.selectbox("Color", COLORS)

    # st.markdown("#### Additional Features")

    col2a, col2b = st.columns(2)

    with col2a:
        air_conditioner = st.checkbox("Air Conditioner", True)
        security_system = st.checkbox("Security System", False)
        parking_sensors = st.checkbox("Parking Sensors", False)

    with col2b:
        customs_cleared = st.checkbox("Customs Cleared", True)
        power_windows = st.checkbox("Power Windows", True)
        power_mirrors = st.checkbox("Power Mirrors", True)


st.markdown("---")


# ---------------------------------------------------------
#  PREDICTION
# ---------------------------------------------------------

if st.button("predict price", type="primary", use_container_width=True):

    if model is None:
        st.error("Model not loaded!")
    else:
        try:
            # Build input df
            input_df = pd.DataFrame({
                "transmission": [transmission],
                "mileage": [mileage],
                "release_year": [release_year],
                "engine_volume": [engine_volume],
                "month_year": [2025],
                # 'car_age':[2026-release_year],
                "brand_type": [brand_type],
                "car_name": [car_name],
                "Air_Conditioner": [int(air_conditioner)],
                "Security_System": [int(security_system)],
                "Parking_Sensors": [int(parking_sensors)],
                "Customs_Cleared": [int(customs_cleared)],
                "Power_Windows": [int(power_windows)],
                "Power_Mirrors": [int(power_mirrors)],
                "fuel_type": [fuel_type],
                "item_type": [item_type],
                "owners_count": [owners_count],
                "car_condition": [car_condition],
                "color": [color],
                "body_type": [body_type],
                "state": [state]
            })

            # Encode categorical columns
            cat_cols = [
                "car_name", "fuel_type", "item_type",
                "owners_count", "car_condition",
                "color", "body_type", "state"
            ]
            # cat_cols = [
            #     "car_name", "fuel_type", "item_type",
            #     "car_condition",
            #     "color", "body_type", "state"
            # ]
            orig_cols = input_df.columns.tolist()
            transformed = encoder.transform(input_df, cat_cols)

            if isinstance(transformed, pd.DataFrame):
                input_df = transformed.copy()
            else:
                # build new column names: replace each categorical col with <col>_tencoded
                new_cols = []
                for c in orig_cols:
                    if c in cat_cols:
                        new_cols.append(f"{c}_tencoded")
                    else:
                        new_cols.append(c)
                # rebuild with correct names
                input_df = pd.DataFrame(transformed, columns=new_cols)
            # Time index create
            # input_df = time_index(input_df)
            input_df.to_csv('testsdad.csv',index=False)

            # Prediction
            try:
                input_df = input_df[model.feature_name_]
            except:
                print('error coming from input df ')
            
            prediction = model.predict(input_df)[0]
            prediction = np.expm1(prediction)
            st.success("✅ Prediction Complete!")

            # Display
            col_r1, col_r2, col_r3 = st.columns(3)

            with col_r1:
                st.metric("prediction", f"{prediction:,.2f}")

            with col_r2:
                st.metric("10% price range", f"{prediction*0.9:,.2f} - {prediction*1.1:,.2f}")

            # with col_r3:
            #     st.metric("", f"{datetime.now().year - release_year} years")

        except Exception as e:
            st.error(f"Error: {e}")


# ---------------------------------------------------------
#  SIDEBAR
# ---------------------------------------------------------

# with st.sidebar:
#     st.header("ℹ️ Instructions")
#     st.markdown("""
#     **How to use this app:**
#     1. Fill all details
#     2. Click **Predict Price**
#     3. See predicted value & range
    
#     **Required Files:**
#     - target_encoder.pkl
#     - model.pkl
#     - Car_State_name.csv
#     """)

