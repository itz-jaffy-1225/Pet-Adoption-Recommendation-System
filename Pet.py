import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Pet Adoption Predictor", layout="wide", page_icon="🐶")

# Title and header with animation
st.markdown(
    """
    <style>
    /* Main Title Animation */
    @keyframes fadeIn {
        0% { opacity: 0; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }
    @keyframes slideIn {
        0% { transform: translateY(-50px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    .main-title {
        font-size: 50px; 
        color: #4CAF50; 
        font-family: 'Arial', sans-serif; 
        text-align: center;
        animation: fadeIn 1.5s ease-in-out;
    }
    .subtitle {
        font-size: 20px; 
        color: #555;
        text-align: center;
        animation: slideIn 1.5s ease-in-out;
    }
    
    /* Button Hover Effect */
    button:hover {
        background-color: #4CAF50 !important;
        color: white !important;
        transform: scale(1.05);
        transition: 0.3s ease-in-out;
    }
    
    /* Success and Error Box Animations */
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    @keyframes shake {
        0% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        50% { transform: translateX(5px); }
        100% { transform: translateX(0); }
    }
    .success-box {
        font-size: 20px; 
        color: #4CAF50; 
        animation: bounce 1s infinite alternate;
    }
    .error-box {
        font-size: 20px; 
        color: #FF5252; 
        animation: shake 0.5s infinite alternate;
    }
    
    /* Sidebar Animation */
    .stSidebar {
        animation: fadeIn 1.5s ease-in-out;
    }
    
    /* Footer Animation */
    @keyframes scroll {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
    }
    .footer .scrolling-text {
        display: inline-block;
        animation: scroll 10s linear infinite;
        white-space: nowrap;
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }
    .loading {
        font-size: 18px;
        color: #FF9800;
        animation: pulse 1.5s infinite;
    }
    </style>

    <div>
        <h1 class="main-title">🐶 Pet Adoption Likelihood Predictor 🐱</h1>
        <p class="subtitle">Find out the chances of a pet being adopted and help them find their forever home! ✨</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("Customize Your Inputs 🛠️")
st.sidebar.write("Adjust the sliders and options below to see predictions!")

# Load dataset
@st.cache_data
def load_data():
    # Replace 'your_dataset.csv' with the path to your dataset
    df = pd.read_csv('pet_adoption_data.csv')
    return df

data = load_data()

# Feature preparation
X = data.drop(['AdoptionLikelihood', 'PetID'], axis=1)
y = data['AdoptionLikelihood']
X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical variables

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)



# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Sidebar user inputs
pet_type = st.sidebar.selectbox("🐾 Pet Type", data['PetType'].unique())

# Filter breed options based on the selected pet type
filtered_breeds = data[data['PetType'] == pet_type]['Breed'].unique()
breed = st.sidebar.selectbox("🦴 Breed", filtered_breeds)

age = st.sidebar.slider("🎂 Age in Months", int(data['AgeMonths'].min()), int(data['AgeMonths'].max()), step=1)
color = st.sidebar.selectbox("🎨 Color", data['Color'].unique())
size = st.sidebar.selectbox("📏 Size", data['Size'].unique())
weight = st.sidebar.slider("⚖️ Weight (kg)", float(data['WeightKg'].min()), float(data['WeightKg'].max()), step=0.1)
vaccinated = st.sidebar.selectbox("💉 Vaccinated", ["Yes", "No"])
health_condition = st.sidebar.selectbox("❤️ Health Condition", ["Healthy", "Needs Attention"])
time_in_shelter = st.sidebar.slider("🏠 Time in Shelter (days)", int(data['TimeInShelterDays'].min()), int(data['TimeInShelterDays'].max()), step=1)
adoption_fee = st.sidebar.slider("💲 Adoption Fee", int(data['AdoptionFee'].min()), int(data['AdoptionFee'].max()), step=1)
previous_owner = st.sidebar.selectbox("👨‍👩‍👧 Previous Owner", ["Yes", "No"])

# Prepare input data
input_data = pd.DataFrame({
    'PetType': [pet_type],
    'Breed': [breed],
    'AgeMonths': [age],
    'Color': [color],
    'Size': [size],
    'WeightKg': [weight],
    'Vaccinated': [1 if vaccinated == "Yes" else 0],
    'HealthCondition': [0 if health_condition == "Healthy" else 1],
    'TimeInShelterDays': [time_in_shelter],
    'AdoptionFee': [adoption_fee],
    'PreviousOwner': [1 if previous_owner == "Yes" else 0]
})

# Apply one-hot encoding for categorical variables
input_data = pd.get_dummies(input_data, columns=['PetType', 'Breed', 'Color', 'Size'], drop_first=True)

# Make sure input data matches the columns of the training data
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Make prediction
adoption_probability = model.predict_proba(input_data)[:, 1][0]
result = model.predict(input_data)[0]

# Display results
st.markdown(
    f"""
    <style>
    .success-box {{
        font-size: 20px; 
        color: #4CAF50; 
        animation: bounce 1s infinite alternate;
    }}
    .error-box {{
        font-size: 20px; 
        color: #FF5252; 
        animation: shake 1s infinite;
    }}
    @keyframes bounce {{
        from {{ transform: translateY(0); }}
        to {{ transform: translateY(-10px); }}
    }}
    @keyframes shake {{
        0% {{ transform: translateX(0); }}
        25% {{ transform: translateX(-5px); }}
        50% {{ transform: translateX(5px); }}
        100% {{ transform: translateX(0); }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Result")
    if result == 1:
        st.markdown(f"<div class='success-box'>🐾 This pet is likely to be adopted! 💚 Adoption Likelihood: {adoption_probability * 100:.2f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='error-box'>⚠️ This pet might not be adopted easily. Adoption Likelihood: {adoption_probability * 100:.2f}%</div>", unsafe_allow_html=True)

with col2:
    st.subheader("Feature Importance")
    feature_importances = pd.DataFrame(
        {
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }
    ).sort_values(by='Importance', ascending=False)
    st.bar_chart(feature_importances.set_index('Feature').head(5))

# Footer with custom background and scrolling text
st.markdown(
    """
    <style>
    footer {
        visibility: hidden;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
    }
    .footer .scrolling-text {
        animation: scroll 10s linear infinite;
        white-space: nowrap;
    }
    @keyframes scroll {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    </style>
    <div class="footer">
        <div class="scrolling-text">Adopt, don’t shop! | Help every pet find a home 🐾 | Donate to local shelters today ❤️</div>
    </div>
    """,
    unsafe_allow_html=True
)
