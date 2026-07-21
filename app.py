# import streamlit as st
# import pandas as pd
# import joblib  
# import datetime
# from streamlit_searchbox import st_searchbox

# model = joblib.load("model.pkl")
# expected_columns = joblib.load("columns.pkl")

# st.title("Car Price Prediction By Sahil❤️")
# st.markdown("Provide the following details")
# current_year = datetime.date.today().year


# car_list = [
#     'ritz', 'sx4', 'ciaz', 'wagon r', 'swift', 'vitara brezza',
#     's cross', 'alto 800', 'ertiga', 'dzire', 'alto k10', 'ignis',
#     '800', 'baleno', 'omni', 'fortuner', 'innova', 'corolla altis',
#     'etios cross', 'etios g', 'etios liva', 'corolla', 'etios gd', 'camry',
#     'land cruiser', 'Royal Enfield Thunder 500', 'UM Renegade Mojave', 'KTM RC200',
#     'Bajaj Dominar 400', 'Royal Enfield Classic 350', 'KTM RC390', 'Hyosung GT250R',
#     'Royal Enfield Thunder 350', 'KTM 390 Duke ', 'Mahindra Mojo XT300', 'Bajaj Pulsar RS200','Royal Enfield Bullet 350', 'Royal Enfield Classic 500', 'Bajaj Avenger 220', 'Bajaj Avenger 150','Honda CB Hornet 160R', 'Yamaha FZ S V 2.0', 'Yamaha FZ 16', 'TVS Apache RTR 160','Bajaj Pulsar 150', 'Honda CBR 150', 'Hero Extreme', 'Bajaj Avenger 220 dtsi','xcent', 'elantra', 'creta', 'verna', 'city', 'brio','amaze', 'jazz'
# ]

# # Search function for the component
# def search_cars(search_term: str):
#     if not search_term:
#         return car_list
#     return [car for car in car_list if search_term.lower() in car.lower()]

# car_name = st_searchbox(
#     search_cars,
#     placeholder="Type car name to search...",
#     key="car_search_box"
# )

# selected_year = st.selectbox(
#     "Select Year",
#     range(current_year, 1989, -1) # Displays newest year first
# )
# Present_price = st.number_input(
#     "Enter Present Price",
#     min_value=0.00,
#     max_value=10000000.00,
#     value=19.99, # Default value
#     step=0.01,   # Increments by cents/paise
#     format="%.2f" # Forces 2 decimal places
# )
# kms_driven = st.number_input(
#     "Kilometers Driven",
#     min_value=0,
#     max_value=500000,
#     value=15000,     # Default average value
#     step=1000        # Increments by 1,000 km per click
# )
# fuel_type = st.selectbox(
#     "Select Fuel Type",
#     ["Petrol", "Diesel", "CNG"]
# )
# seller_type = st.selectbox(
#     "Select Seller Type",
#     ["Dealer", "Individual"]
# )
# transmission = st.selectbox(
#     "Select Transmission Type",
#     ["Automatic", "Manual"]
# )
# owner_value = st.selectbox(
#     "Select Owner Type Value",
#     options=[0, 1, 2],
#     format_func=lambda x: f"First Owner" if x == 0 else (f"Second Owner" if x == 1 else f"Third+ Owner")
# )

# # ... (your existing code ends here)

# # 1. Create a submit button
# if st.button("Predict Selling Price"):
    
#     # 2. Calculate the 'Age' feature if your model used it instead of raw year
#     # (Common in this dataset: Year is often converted to age_of_car = current_year - selected_year)
#     car_age = current_year - selected_year

#     # 3. Create a raw dictionary from user inputs
#     # Ensure these keys match the exact column names your model trained on
#     input_data = {
#         "Year": selected_year,
#         "Present_Price": Present_price,
#         "Kms_Driven": kms_driven,
#         "Fuel_Type": fuel_type,
#         "Seller_Type": seller_type,
#         "Transmission": transmission,
#         "Owner": owner_value
#     }
    
#     # Add 'Age' to dictionary if your dataset used it instead of 'Year'
#     # input_data["Age"] = car_age 

#     # 4. Convert inputs into a Pandas DataFrame
#     input_df = pd.DataFrame([input_data])

#     # 5. Handle Categorical Encoding (One-Hot Encoding)
#     # This creates the dummy variables (e.g., Fuel_Type_Diesel, Seller_Type_Individual)
#     input_df_encoded = pd.get_dummies(input_df)

#     # 6. Align columns with the training dataset
#     # Reindex fills missing encoded columns with 0 and drops unneeded ones (like Car Name)
#     final_features = input_df_encoded.reindex(columns=expected_columns, fill_value=0)

#     # 7. Make the prediction using your loaded joblib model
#     try:
#         prediction = model.predict(final_features)[0]
        
#         # 8. Display the result beautifully
#         st.success(f"### Predicted Selling Price: ₹ {prediction:.2f} Lakhs")
#         st.balloons()
        
#     except Exception as e:
#         st.error(f"Prediction failed. Error details: {e}")
#         st.info("Tip: Double-check that your inputs match the features inside 'columns.pkl'.")

import streamlit as st
import pandas as pd
import joblib
import datetime
import time
from streamlit_searchbox import st_searchbox

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resource loading (only loads once, not on every rerun)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load("model.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, expected_columns

model, expected_columns = load_artifacts()

current_year = datetime.date.today().year

CAR_LIST = [
    'ritz', 'sx4', 'ciaz', 'wagon r', 'swift', 'vitara brezza',
    's cross', 'alto 800', 'ertiga', 'dzire', 'alto k10', 'ignis',
    '800', 'baleno', 'omni', 'fortuner', 'innova', 'corolla altis',
    'etios cross', 'etios g', 'etios liva', 'corolla', 'etios gd', 'camry',
    'land cruiser', 'Royal Enfield Thunder 500', 'UM Renegade Mojave', 'KTM RC200',
    'Bajaj Dominar 400', 'Royal Enfield Classic 350', 'KTM RC390', 'Hyosung GT250R',
    'Royal Enfield Thunder 350', 'KTM 390 Duke ', 'Mahindra Mojo XT300', 'Bajaj Pulsar RS200',
    'Royal Enfield Bullet 350', 'Royal Enfield Classic 500', 'Bajaj Avenger 220', 'Bajaj Avenger 150',
    'Honda CB Hornet 160R', 'Yamaha FZ S V 2.0', 'Yamaha FZ 16', 'TVS Apache RTR 160',
    'Bajaj Pulsar 150', 'Honda CBR 150', 'Hero Extreme', 'Bajaj Avenger 220 dtsi',
    'xcent', 'elantra', 'creta', 'verna', 'city', 'brio', 'amaze', 'jazz'
]

def search_cars(search_term: str):
    if not search_term:
        return CAR_LIST
    return [car for car in CAR_LIST if search_term.lower() in car.lower()]

# ---------------------------------------------------------------------------
# Session state — prediction history
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: car, year, present_price, predicted

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.stApp {
    background: linear-gradient(-45deg, #1a1a2e, #16213e, #23233d, #1f1a3d);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.main .block-container { animation: fadeUp 0.8s ease-out; padding-top: 2rem; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

.hero-title {
    text-align: center;
    font-size: 2.9rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    animation: popIn 0.7s ease-out;
}

.hero-title .title-gradient {
    background: linear-gradient(90deg, #ff9a3c, #ff6a00, #ff9a3c);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
    animation: shine 4s linear infinite;
}

@keyframes shine { to { background-position: 200% center; } }

@keyframes popIn {
    0% { opacity: 0; transform: scale(0.85); }
    60% { opacity: 1; transform: scale(1.03); }
    100% { transform: scale(1); }
}

.car-icon { display: inline-block; animation: drive 2.6s ease-in-out infinite; margin-right: 0.35rem; }

@keyframes drive {
    0%, 100% { transform: translateX(0) rotate(0deg); }
    25% { transform: translateX(4px) rotate(-2deg); }
    75% { transform: translateX(-4px) rotate(2deg); }
}

.hero-subtitle {
    text-align: center;
    color: #cfd3ea;
    font-size: 1.05rem;
    font-weight: 300;
    margin-bottom: 1.6rem;
    animation: fadeUp 1s ease-out;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 18px !important;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.2rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border 0.25s ease;
    animation: fadeUp 0.9s ease-out;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 12px 28px rgba(255, 106, 0, 0.18);
    border: 1px solid rgba(255, 106, 0, 0.35) !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16213e 0%, #1f1a3d 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
    color: #e6e6f0 !important;
}

label, .stMarkdown p, .stMarkdown li { color: #e8e8f2 !important; }

div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #ff9a3c, #ff6a00);
    background-size: 200% auto;
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.8rem 1.1rem;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    box-shadow: 0 6px 20px rgba(255, 106, 0, 0.35);
    transition: all 0.35s ease;
}

div.stButton > button:hover:not(:disabled) {
    background-position: right center;
    transform: scale(1.02) translateY(-2px);
    box-shadow: 0 10px 28px rgba(255, 106, 0, 0.55);
}

div.stButton > button:disabled {
    opacity: 0.45;
    box-shadow: none;
}

div.stButton > button:active { transform: scale(0.98); }

.section-header {
    color: #ffb37a;
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 0.6rem;
    letter-spacing: 0.3px;
}

/* Loading road animation */
.road-wrap {
    position: relative;
    height: 50px;
    background: repeating-linear-gradient(
        90deg, #444 0px, #444 30px, transparent 30px, transparent 55px
    );
    border-radius: 8px;
    overflow: hidden;
    margin: 0.8rem 0;
}

.road-car {
    position: absolute;
    top: 6px;
    font-size: 1.8rem;
    animation: roadDrive 1.4s linear infinite;
}

@keyframes roadDrive {
    from { left: -10%; }
    to { left: 105%; }
}

/* Result card */
.result-card {
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    animation: resultPop 0.6s cubic-bezier(0.26, 1.36, 0.44, 1);
    margin-top: 1rem;
    background: linear-gradient(135deg, rgba(255, 154, 60, 0.18), rgba(255, 106, 0, 0.08));
    border: 1px solid rgba(255, 154, 60, 0.4);
    box-shadow: 0 0 40px rgba(255, 106, 0, 0.22);
}

@keyframes resultPop {
    0% { opacity: 0; transform: scale(0.8) translateY(15px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

.price-icon { font-size: 2.6rem; animation: bounceIn 0.8s ease-out; }

@keyframes bounceIn {
    0% { transform: scale(0); }
    60% { transform: scale(1.25); }
    100% { transform: scale(1); }
}

.price-value {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ffb37a, #ff6a00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 0.3rem;
    animation: popIn 0.5s ease-out;
}

.result-sub { color: #d8d8e6; font-size: 0.92rem; margin-top: 0.4rem; }

.depre-bar-wrap {
    margin-top: 1.2rem;
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}

.depre-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #ff6a00, #ffb37a);
    animation: fillBar 1s ease-out;
}

@keyframes fillBar {
    from { width: 0%; }
}

.chip-row { display: flex; gap: 0.6rem; flex-wrap: wrap; justify-content: center; margin-top: 1.2rem; }

.chip {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font-size: 0.82rem;
    color: #e6e6f0;
    animation: chipIn 0.5s ease-out backwards;
}

.chip:nth-child(1){animation-delay:0.05s;}
.chip:nth-child(2){animation-delay:0.12s;}
.chip:nth-child(3){animation-delay:0.19s;}
.chip:nth-child(4){animation-delay:0.26s;}
.chip:nth-child(5){animation-delay:0.33s;}
.chip:nth-child(6){animation-delay:0.40s;}

@keyframes chipIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.history-item {
    border-left: 3px solid #ff6a00;
    padding: 0.5rem 0.8rem;
    margin-bottom: 0.5rem;
    border-radius: 8px;
    background: rgba(255,255,255,0.04);
    font-size: 0.85rem;
    animation: fadeUp 0.4s ease-out;
}

.stat-chip {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 0.6rem 0.8rem;
    text-align: center;
}

.stat-num { font-size: 1.3rem; font-weight: 700; color: #ff9a3c; }
.stat-label { font-size: 0.72rem; color: #b3a9d8; }

.footer-note {
    text-align: center;
    color: #8f8fae;
    font-size: 0.78rem;
    margin-top: 2.5rem;
    animation: fadeUp 1.2s ease-out;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚗 About this tool")
    st.markdown(
        "Estimates a used vehicle's resale price using a trained "
        "**regression model** based on its age, mileage, fuel type, "
        "and ownership history."
    )
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    total = len(st.session_state.history)
    avg_price = (
        sum(h["predicted"] for h in st.session_state.history) / total
        if total else 0
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='stat-chip'><div class='stat-num'>{total}</div><div class='stat-label'>Estimates</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='stat-chip'><div class='stat-num'>₹{avg_price:.1f}L</div><div class='stat-label'>Avg. Price</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.history:
        st.markdown("### 🕓 Recent Estimates")
        for h in reversed(st.session_state.history[-6:]):
            st.markdown(
                f"<div class='history-item'>🚙 {h['car'].title()} ({h['year']}) "
                f"→ ₹{h['predicted']:.2f}L</div>",
                unsafe_allow_html=True,
            )
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown(
            "<span style='font-size:0.82rem;color:#9a9ac0;'>No estimates yet.</span>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='hero-title'><span class='car-icon'>🚗</span>"
    "<span class='title-gradient'>Car Price Prediction</span></div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='hero-subtitle'>Crafted by Sahil — fill in your car's "
    "details for an instant resale price estimate</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    card1 = st.container(border=True)
    with card1:
        st.markdown("<div class='section-header'>🚘 Vehicle Details</div>", unsafe_allow_html=True)
        car_name = st_searchbox(
            search_cars,
            placeholder="Type car name to search...",
            key="car_search_box",
        )
        selected_year = st.selectbox("Select Year", range(current_year, 1989, -1))
        present_price = st.number_input(
            "Enter Present Price (₹ Lakhs)",
            min_value=0.00,
            max_value=10000000.00,
            value=19.99,
            step=0.01,
            format="%.2f",
        )
        kms_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=500000,
            value=15000,
            step=1000,
        )

with col2:
    card2 = st.container(border=True)
    with card2:
        st.markdown("<div class='section-header'>⚙️ Ownership & Type</div>", unsafe_allow_html=True)
        fuel_type = st.selectbox("Select Fuel Type", ["Petrol", "Diesel", "CNG"])
        seller_type = st.selectbox("Select Seller Type", ["Dealer", "Individual"])
        transmission = st.selectbox("Select Transmission Type", ["Automatic", "Manual"])
        owner_value = st.selectbox(
            "Select Owner Type Value",
            options=[0, 1, 2],
            format_func=lambda x: "First Owner" if x == 0 else ("Second Owner" if x == 1 else "Third+ Owner"),
        )

car_age = current_year - selected_year
st.markdown(
    f"<div style='text-align:center;font-size:0.85rem;color:#b3a9d8;margin-top:-0.4rem;'>"
    f"🕓 Car age: {car_age} year{'s' if car_age != 1 else ''}</div>",
    unsafe_allow_html=True,
)

st.write("")
predict_disabled = not car_name
if predict_disabled:
    st.markdown(
        "<div style='text-align:center;font-size:0.82rem;color:#ffb37a;margin-bottom:0.5rem;'>"
        "👆 Please select a car name above to enable prediction</div>",
        unsafe_allow_html=True,
    )
predict_clicked = st.button("💰 Predict Selling Price", disabled=predict_disabled)

# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
if predict_clicked:

    road_placeholder = st.empty()
    road_placeholder.markdown("""
    <div class="road-wrap"><div class="road-car">🚗</div></div>
    """, unsafe_allow_html=True)
    time.sleep(0.7)
    road_placeholder.empty()

    input_data = {
        "Year": selected_year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
        "Owner": owner_value,
    }

    input_df = pd.DataFrame([input_data])
    input_df_encoded = pd.get_dummies(input_df)
    final_features = input_df_encoded.reindex(columns=expected_columns, fill_value=0)

    try:
        prediction = model.predict(final_features)[0]
        prediction = max(prediction, 0)  # guard against negative predictions

        st.session_state.history.append({
            "car": car_name,
            "year": selected_year,
            "present_price": present_price,
            "predicted": prediction,
        })

        depreciation_pct = (
            max(0, (present_price - prediction) / present_price * 100)
            if present_price > 0 else 0
        )

        st.markdown(f"""
        <div class="result-card">
            <div class="price-icon">💰</div>
            <div class="price-value">₹ {prediction:.2f} Lakhs</div>
            <div class="result-sub">Estimated selling price for your {car_name.title()}</div>
            <div class="depre-bar-wrap">
                <div class="depre-bar-fill" style="width:{min(depreciation_pct, 100):.1f}%;"></div>
            </div>
            <div class="result-sub">📉 {depreciation_pct:.1f}% depreciation from present price</div>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

        st.markdown(f"""
        <div class="chip-row">
            <div class="chip">🚙 {car_name.title()}</div>
            <div class="chip">🕓 {car_age} yrs old</div>
            <div class="chip">🛣️ {kms_driven:,} km</div>
            <div class="chip">⛽ {fuel_type}</div>
            <div class="chip">⚙️ {transmission}</div>
            <div class="chip">👤 {['First', 'Second', 'Third+'][owner_value]} Owner</div>
        </div>
        """, unsafe_allow_html=True)

        # Downloadable report
        report_text = (
            f"Car Price Prediction Report\n"
            f"----------------------------\n"
            f"Car: {car_name.title()}\n"
            f"Year: {selected_year} (Age: {car_age} yrs)\n"
            f"Present Price: Rs {present_price:.2f} Lakhs\n"
            f"Kilometers Driven: {kms_driven:,} km\n"
            f"Fuel Type: {fuel_type}\n"
            f"Seller Type: {seller_type}\n"
            f"Transmission: {transmission}\n"
            f"Owner: {['First', 'Second', 'Third+'][owner_value]} Owner\n"
            f"----------------------------\n"
            f"Predicted Selling Price: Rs {prediction:.2f} Lakhs\n"
            f"Depreciation: {depreciation_pct:.1f}%\n"
        )
        st.download_button(
            "📄 Download report",
            data=report_text,
            file_name=f"{car_name.replace(' ', '_')}_price_report.txt",
            mime="text/plain",
        )

    except Exception as e:
        st.error(f"Prediction failed. Error details: {e}")
        st.info("Tip: Double-check that your inputs match the features inside 'columns.pkl'.")

st.markdown(
    "<div class='footer-note'>Built with Streamlit · Regression Model · Made with ❤️ by Sahil</div>",
    unsafe_allow_html=True,
)