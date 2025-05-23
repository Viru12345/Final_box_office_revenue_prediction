
import streamlit as st
import numpy as np
import joblib

# Load models
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
vectorizer = joblib.load("vectorizer.pkl")

distributors = ["Disney", "Warner Bros.", "Paramount", "Sony", "Universal"]
mpaa_ratings = ["G", "PG", "PG-13", "R", "NC-17"]

st.title("🎬 Box Office Revenue Predictor")

distributor = st.selectbox("Distributor", distributors)
mpaa = st.selectbox("MPAA Rating", mpaa_ratings)
genres = st.text_input("Genres (e.g., Action Adventure)", "Action")
theaters = st.number_input("Opening Theaters", 100, 7000, 3500)
days = st.number_input("Release Days", 1, 365, 120)

if st.button("Predict"):
    distributor_encoded = distributors.index(distributor)
    mpaa_encoded = mpaa_ratings.index(mpaa)
    genre_vector = vectorizer.transform([genres]).toarray()[0]

    input_vec = [distributor_encoded, mpaa_encoded] + list(genre_vector) + [
        np.log10(theaters + 1), np.log10(days + 1)
    ]

    expected = model.n_features_in_
    if len(input_vec) < expected:
        input_vec += [0] * (expected - len(input_vec))
    elif len(input_vec) > expected:
        input_vec = input_vec[:expected]

    input_vec = np.array(input_vec).reshape(1, -1)
    input_scaled = scaler.transform(input_vec)
    log_revenue = model.predict(input_scaled)[0]
    revenue = 10 ** log_revenue - 1

    st.success(f"📊 Predicted Revenue: ${revenue:,.2f}")
