# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# import pandas as pd
# import pickle

# # Load the trained model
# model = tf.keras.models.load_model('05_model.h5')

# # Load the encoders and scaler
# with open('02_label_encoder_gender.pkl', 'rb') as file:
#     label_encoder_gender = pickle.load(file)

# with open('03_onehot_encoder_geo.pkl', 'rb') as file:
#     onehot_encoder_geo = pickle.load(file)

# with open('04_scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)


# ## streamlit app
# st.title('Customer Churn PRediction')

# # User input
# geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
# gender = st.selectbox('Gender', label_encoder_gender.classes_)
# age = st.slider('Age', 18, 92)
# balance = st.number_input('Balance')
# credit_score = st.number_input('Credit Score')
# estimated_salary = st.number_input('Estimated Salary')
# tenure = st.slider('Tenure', 0, 10)
# num_of_products = st.slider('Number of Products', 1, 4)
# has_cr_card = st.selectbox('Has Credit Card', [0, 1])
# is_active_member = st.selectbox('Is Active Member', [0, 1])

# # Prepare the input data
# input_data = pd.DataFrame({
#     'CreditScore': [credit_score],
#     'Gender': [label_encoder_gender.transform([gender])[0]],
#     'Age': [age],
#     'Tenure': [tenure],
#     'Balance': [balance],
#     'NumOfProducts': [num_of_products],
#     'HasCrCard': [has_cr_card],
#     'IsActiveMember': [is_active_member],
#     'EstimatedSalary': [estimated_salary]
# })

# # One-hot encode 'Geography'
# geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
# geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# # Combine one-hot encoded columns with input data
# input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# # Scale the input data
# input_data_scaled = scaler.transform(input_data)


# # Predict churn
# prediction = model.predict(input_data_scaled)
# prediction_proba = prediction[0][0]

# st.write(f'Churn Probability: {prediction_proba:.2f}')

# if prediction_proba > 0.5:
#     st.write('The customer is likely to churn.')
# else:
#     st.write('The customer is not likely to churn.')

import streamlit as st

# Set page configuration at the top before any other Streamlit commands
st.set_page_config(
    page_title="Customer Churn Prediction", 
    layout="wide"
)

import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

import warnings
warnings.filterwarnings("ignore")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('05_model.h5')

model = load_model()

# Load encoders and scaler
@st.cache_resource
def load_encoders():
    with open('02_label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('03_onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('04_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    return label_encoder_gender, onehot_encoder_geo, scaler

label_encoder_gender, onehot_encoder_geo, scaler = load_encoders()

st.title('🔍 Customer Churn Prediction')
st.markdown("Predict whether a customer is likely to churn based on their profile and activity.")

# Sidebar for user input
st.sidebar.title("📌 :red[Customer Details]")

geography = st.sidebar.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('⚤ Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('🎂 Age', 18, 92, 30)
balance = st.sidebar.number_input('💰 Balance', min_value=0.0, value=0.0, step=1000.0)
credit_score = st.sidebar.number_input('📊 Credit Score', min_value=300, max_value=900, value=650)
estimated_salary = st.sidebar.number_input('💵 Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)
tenure = st.sidebar.slider('📅 Tenure (Years)', 0, 10)
num_of_products = st.sidebar.slider('📦 Number of Products', 1, 4)
has_cr_card = st.sidebar.radio('💳 Has Credit Card?', [0, 1], format_func=lambda x: "Yes" if x else "No")
is_active_member = st.sidebar.radio('✅ Is Active Member?', [0, 1], format_func=lambda x: "Yes" if x else "No")

# Process input data
input_data = np.array([[credit_score, label_encoder_gender.transform([gender])[0], age, tenure, balance, 
                         num_of_products, has_cr_card, is_active_member, estimated_salary]])

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
input_data = np.hstack([input_data, geo_encoded])

# if gender not in label_encoder_gender.classes_:
#     st.error(f"Error: '{gender}' is not a recognized gender. Available options: {label_encoder_gender.classes_}")
# else:
#     gender_encoded = label_encoder_gender.transform([gender])[0]

# if gender not in label_encoder_gender.classes_:
#     st.error(f"Invalid gender: {gender}. Expected: {label_encoder_gender.classes_}")
#     st.stop()
# gender_encoded = label_encoder_gender.transform([gender])[0]

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)[0][0]

# Display prediction
st.subheader("📝 Prediction Result")
st.write(f"**Churn Probability: {prediction:.2%}**")

if prediction > 0.5:
    st.error("⚠️ The customer is **likely to churn**.")
else:
    st.success("✅ The customer is **not likely to churn**.")

# Add a progress bar to indicate risk level
st.progress(min(int(prediction * 100), 100))

# Display helpful insights
st.markdown("---")
st.markdown("📌 **Insights & Recommendations:**")
if prediction > 0.7:
    st.warning("🚨 High churn risk! Consider offering loyalty programs, personalized offers, or improving customer engagement.")
elif prediction > 0.5:
    st.info("⚠️ Moderate risk. A proactive approach can help retain the customer.")
else:
    st.success("✅ Low risk. Continue maintaining a positive customer experience.")