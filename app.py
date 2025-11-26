import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from Target_encoder import TargetEncoder

# feature pricing
feature_prices = {
    'Air_Conditioner': 50,
    'Security_System': 40,
    'Parking_Sensors': 30,
    'Customs_Cleared': 100,
    'Power_Windows': 25,
    'Power_Mirrors': 20
}

@st.cache_data
def load_names():
    try:
        return pd.read_csv("Car_State_name.csv")
    except FileNotFoundError:
        st.error("car_State_name.csv file not found")
        return pd.DataFrame()

@st.cache_resource
def load_encoder():
    try:
        with open("target_encoder_extra.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("encoder file not found")
        return None

@st.cache_resource
def load_model():
    try:
        with open("CHEVROLET_DAEWOO_RAVON_LGBM_extra.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("model file not found")
        return None

all_names = load_names()
encoder = load_encoder()
model = load_model()

st.set_page_config(page_title="car price predictor", layout="wide")
st.title("car price prediction")
st.markdown("---")

if all_names.empty or encoder is None or model is None:
    st.error("required files not loaded. check files exist.")
    st.stop()

car_names = sorted(all_names["car_name"].unique().tolist())
regions = sorted(all_names["state"].unique().tolist())
fuel_types = sorted(all_names["fuel_type"].unique().tolist())
item_types = sorted(all_names["item_type"].unique().tolist())
body_types = sorted(all_names["body_type"].unique().tolist())
colors = sorted(all_names["color"].unique().tolist())
car_conditions = sorted(all_names["car_condition"].unique().tolist())
owners_count = ["1", "2", "3"]

col1, col2 = st.columns(2)

with col1:
    st.subheader("basic information")
    
    default_car = "Cobalt" if "Cobalt" in car_names else car_names[0]
    car_name = st.selectbox("car name", car_names, index=car_names.index(default_car))
    state = st.selectbox("region", regions)
    release_year = st.number_input("release year", min_value=1990, max_value=datetime.now().year, value=2020, step=1)
    transmission = st.radio("transmission", options=[0, 1], index=0, format_func=lambda x: "automatic" if x == 0 else "manual", horizontal=True)
    mileage = st.number_input("mileage (km)", min_value=0.0, max_value=500000.0, value=80000.0, step=1000.0)
    engine_volume = st.number_input("engine volume (L)", min_value=0.5, max_value=8.0, value=1.5, step=0.1)
    brand_type = st.selectbox("brand type", options=[1, 0], format_func=lambda x: "chevrolet" if x == 1 else "daewoo")

with col2:
    st.subheader("details")
    fuel_type = st.selectbox("fuel type", fuel_types)
    item_type = st.selectbox("item type", item_types)
    default_body = "Sedan" if "Sedan" in body_types else body_types[0]
    body_type = st.selectbox("body type", body_types, index=body_types.index(default_body))
    owners_count_sel = st.selectbox("previous owners", owners_count)
    car_condition = st.selectbox("car condition", car_conditions)
    default_color = "White" if "White" in colors else colors[0]
    color = st.selectbox("color", colors, index=colors.index(default_color))
    
    st.markdown("#### additional features")
    col2a, col2b = st.columns(2)
    with col2a:
        air_conditioner = st.checkbox("air conditioner", True)
        security_system = st.checkbox("security system", False)
        parking_sensors = st.checkbox("parking sensors", False)
    with col2b:
        customs_cleared = st.checkbox("customs cleared", True)
        power_windows = st.checkbox("power windows", True)
        power_mirrors = st.checkbox("power mirrors", True)

st.markdown("---")

if st.button("predict price", type="primary"):
    with st.spinner("calculating..."):
        try:
            input_data = {
                "transmission": [transmission],
                "mileage": [mileage],
                "release_year": [release_year],
                "engine_volume": [engine_volume],
                "year": [2025],
                "brand_type": [brand_type],
                "car_name": [car_name],
                "fuel_type": [fuel_type],
                "item_type": [item_type],
                "owners_count": [owners_count_sel],
                "car_condition": [car_condition],
                "color": [color],
                "body_type": [body_type],
                "state": [state]
            }
            
            input_df = pd.DataFrame(input_data)
            cat_cols = ['car_name', 'body_type', 'fuel_type', 'item_type', 'car_condition', 'color', 'state']
            orig_cols = input_df.columns.tolist()
            transformed = encoder.transform(input_df, cat_cols)

            if isinstance(transformed, pd.DataFrame):
                input_df_encoded = transformed.copy()
            else:
                new_cols = [f"{c}_tencoded" if c in cat_cols else c for c in orig_cols]
                input_df_encoded = pd.DataFrame(transformed, columns=new_cols)

            try:
                expected_features = model.feature_name_
                for col in expected_features:
                    if col not in input_df_encoded.columns:
                        input_df_encoded[col] = 0
                input_df_encoded = input_df_encoded[expected_features]
            except AttributeError:
                st.warning("model doesnt have feature_name_ attribute")
            
            if "owners_count" in input_df_encoded.columns:
                input_df_encoded["owners_count"] = input_df_encoded["owners_count"].astype(int)

            raw_prediction = model.predict(input_df_encoded)[0]
            
            # check if log transformed
            if raw_prediction < 10:
                base_price = np.expm1(raw_prediction)
            else:
                base_price = raw_prediction
            
            if not np.isfinite(base_price) or base_price < 0:
                st.error(f"invalid prediction: {base_price}")
                st.info("model produced invalid result")
                st.write(f"raw prediction: {raw_prediction}")
                st.stop()
            
            selected_features = {
                'Air_Conditioner': air_conditioner,
                'Security_System': security_system,
                'Parking_Sensors': parking_sensors,
                'Customs_Cleared': customs_cleared,
                'Power_Windows': power_windows,
                'Power_Mirrors': power_mirrors
            }
            
            feature_breakdown = {}
            total_feature_value = 0
            
            for feature_name, is_selected in selected_features.items():
                if is_selected:
                    feature_price = feature_prices.get(feature_name, 0)
                    feature_breakdown[feature_name] = feature_price
                    total_feature_value += feature_price
            
            price_with_features = base_price + total_feature_value
            final_prediction = price_with_features * 0.85

            st.success("prediction complete")
            col_r1, col_r2, col_r3 = st.columns(3)

            with col_r1:
                st.metric("predicted price", f"${final_prediction:,.2f}")
            with col_r2:
                range_low = final_prediction * 0.9
                range_high = final_prediction * 1.1
                st.metric("price range", f"${final_prediction:,.2f}")
            with col_r3:
                car_age = datetime.now().year - release_year
                depreciation = ((car_age / 10) * 100) if car_age <= 10 else 100
                st.metric("car age", f"{car_age} years")
            
            # with st.expander("price breakdown"):
            #     st.write(f"base model prediction: ${base_price:,.2f}")
            #     st.write("")
            #     if feature_breakdown:
            #         st.write("additional features:")
            #         for feature, value in feature_breakdown.items():
            #             feature_display = feature.replace('_', ' ').lower()
            #             st.write(f"  {feature_display}: +${value:,.2f}")
            #         st.write("")
            #         st.write(f"total features value: +${total_feature_value:,.2f}")
            #     else:
            #         st.write("additional features: none")
            #     st.write("")
            #     st.markdown("---")
            #     st.write(f"subtotal: ${price_with_features:,.2f}")
            #     st.write(f"discount (10%): -${price_with_features * 0.1:,.2f}")
            #     st.markdown(f"### final price: ${final_prediction:,.2f}")
            
            with st.expander("debug info"):
                st.write(f"raw model output: {raw_prediction:.2f}")
                st.write(f"input shape: {input_df_encoded.shape}")
                st.dataframe(input_df_encoded.head())
            
            input_df_encoded.to_csv('sample_prediction.csv', index=False)

        except Exception as e:
            st.error(f"prediction error: {str(e)}")
            with st.expander("error details"):
                import traceback
                st.code(traceback.format_exc())
                st.write("possible causes:")
                st.write("- feature mismatch")
                st.write("- corrupted files")
                st.write("- invalid input")

st.markdown("---")

# st.markdown("<div style='text-align: center; color: gray;'><p>car price predictor | built with streamlit</p></div>", unsafe_allow_html=True)

