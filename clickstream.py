import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = pickle.load(open("clickst2.pkl", 'rb'))

# Set up the Streamlit app
st.title('Online Shopping Website')

# Create input fields for the variables
day = st.number_input('Day', min_value=1, max_value=31, step=1)
order = st.text_input('Order')
country = st.selectbox('Country', [
    'Australia', 'Austria', 'Belgium', 'British Virgin Islands', 'Cayman Islands',
    'Christmas Island', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark',
    'Estonia', 'unidentified', 'Faroe Islands', 'Finland', 'France', 'Germany',
    'Greece', 'Hungary', 'Iceland', 'India', 'Ireland', 'Italy', 'Latvia',
    'Lithuania', 'Luxembourg', 'Mexico', 'Netherlands', 'Norway', 'Poland',
    'Portugal', 'Romania', 'Russia', 'San Marino', 'Slovakia', 'Slovenia',
    'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Arab Emirates',
    'United Kingdom', 'USA', 'biz (*.biz)', 'com (*.com)', 'int (*.int)',
    'net (*.net)', 'org (*.org)'
])
main_category = st.selectbox('Main Category', ['trousers', 'skirts', 'blouses', 'sale'])
clothing_model = st.text_input('Clothing Model')
colour = st.selectbox('Colour', [
    'beige', 'black', 'blue', 'brown', 'burgundy', 'gray', 'green', 'navy blue',
    'of many colors', 'olive', 'pink', 'red', 'violet', 'white'
])
location = st.selectbox('Location', ['top left', 'top in the middle', 'top right', 'bottom left', 'bottom in the middle', 'bottom right'])
model_photography = st.selectbox('Model Photography', ['en face', 'profile'])
#Price = st.number_input('Price', step=0.01)


# Perform label encoding on categorical variables
encoder = LabelEncoder()

# Fit and transform country
countries = ['Australia', 'Austria', 'Belgium', 'British Virgin Islands', 'Cayman Islands',
    'Christmas Island', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark',
    'Estonia', 'unidentified', 'Faroe Islands', 'Finland', 'France', 'Germany',
    'Greece', 'Hungary', 'Iceland', 'India', 'Ireland', 'Italy', 'Latvia',
    'Lithuania', 'Luxembourg', 'Mexico', 'Netherlands', 'Norway', 'Poland',
    'Portugal', 'Romania', 'Russia', 'San Marino', 'Slovakia', 'Slovenia',
    'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Arab Emirates',
    'United Kingdom', 'USA', 'biz (*.biz)', 'com (*.com)', 'int (*.int)',
    'net (*.net)', 'org (*.org)']
encoder.fit(countries)
country_encoded = encoder.transform([country])

# Fit and transform main category
main_categories = ['trousers', 'skirts', 'blouses', 'sale']
encoder.fit(main_categories)
main_category_encoded = encoder.transform([main_category])

# Fit and transform colour
colours = ['beige', 'black', 'blue', 'brown', 'burgundy', 'gray', 'green', 'navy blue',
    'of many colors', 'olive', 'pink', 'red', 'violet', 'white']
encoder.fit(colours)
colour_encoded = encoder.transform([colour])

# Fit and transform location
locations = ['top left', 'top in the middle', 'top right', 'bottom left', 'bottom in the middle', 'bottom right']
encoder.fit(locations)
location_encoded = encoder.transform([location])

# Fit and transform model photography
model_photographies = ['en face', 'profile']
encoder.fit(model_photographies)
model_photography_encoded = encoder.transform([model_photography])

# Fit and transform price2
#price2_values = ['yes', 'no']
#encoder.fit(price2_values)
#price2_encoded = encoder.transform([price2])

# Make predictions when the user clicks the "Predict" button
if st.button('Predict'):
    # Create a DataFrame with the input data
    data = pd.DataFrame({
        'DAY': [day],
        'ORDER': [order],
        'COUNTRY': [country_encoded],
        'PAGE 1 (MAIN CATEGORY)': [main_category_encoded],
        'PAGE 2 (CLOTHING MODEL)': [clothing_model],
        'COLOUR': [colour_encoded],
        'LOCATION': [location_encoded],
        'MODEL PHOTOGRAPHY': [model_photography_encoded],
        #'PRICE': [price]
    })

    # Make predictions
    prediction = model.predict(data)
    result = prediction[0]

    st.write("Predicted Price:", result)
    video_url = "https://youtu.be/E1KdSvDnq6M"  # Replace with your YouTube video URL or ID
st.video(video_url)
