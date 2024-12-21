import streamlit as st
import joblib
import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
best_model = joblib.load(r"C:\Users\prasa\OneDrive\Desktop\ZOHO\best_model.pkl")
scaler = joblib.load(r"C:\Users\prasa\OneDrive\Desktop\ZOHO\scaler.pkl")  

# Sentiment analysis function
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    return 1 if sentiment > 0 else 0

# Streamlit UI
st.title('Movie Audience Rating Prediction')

movie_name = st.text_area("Enter the Movie Name 🎥 : ")

rating = st.number_input("Rating", min_value=0, max_value=5, value=2)
genre = st.selectbox("Genre", options=list(range(17)), index=5)
runtime_in_minutes = st.number_input("Runtime in Minutes", min_value=30, max_value=300, value=120)
tomatometer_status = st.selectbox("Tomatometer Status", [0, 1, 2])
tomatometer_rating = st.number_input("Tomatometer Rating", min_value=0, max_value=100, value=75)
tomatometer_count = st.number_input("Tomatometer Count", min_value=0, max_value=1000, value=100)
movie_year = st.number_input("Movie Year", min_value=1900, max_value=2024, value=2020)
date_difference = st.number_input("Date Difference (in days)", min_value=0, max_value=365, value=120)
user_input = st.text_area("Please enter your movie review:")

if all([rating, genre, runtime_in_minutes, tomatometer_status, tomatometer_rating, tomatometer_count, movie_year, date_difference, user_input]):
    review_sentiment = analyze_sentiment(user_input)

    sample_data = pd.DataFrame({
        'rating': [rating],
        'genre': [genre],
        'runtime_in_minutes': [runtime_in_minutes],
        'tomatometer_status': [tomatometer_status],
        'tomatometer_rating': [tomatometer_rating],
        'tomatometer_count': [tomatometer_count],
        'movie_year': [movie_year],
        'date_difference': [date_difference],
        'review_sentiment': [review_sentiment]
    })

    continuous_columns = [
        'runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count',
        'movie_year', 'date_difference'
    ]
    sample_data[continuous_columns] = scaler.transform(sample_data[continuous_columns])

    y_pred = best_model.predict(sample_data)

    st.write(f"Predicted Audience Rating for the movie {movie_name} : {y_pred[0]:.2f} %")

else:
    st.write("Please fill in all the fields before submitting.")
