import streamlit as st
import pandas as pd
import pickle
import os
import random

# Set page config for a cleaner look
st.set_page_config(layout="centered", page_title="Spotify Recommendation")

# --- Heading and Image ---
st.title("Spotify Music Recommendation System")

try:
    st.image("image.jpg", caption="Discover new music clusters!")
except FileNotFoundError:
    st.warning("image.jpg not found. Please place the image file in the same directory as the script.")

st.write("---")

# --- Initialize Session State ---
if 'predicted_cluster' not in st.session_state:
    st.session_state.predicted_cluster = None
if 'recommendations_data' not in st.session_state:
    st.session_state.recommendations_data = pd.DataFrame(columns=['track_name', 'artist_name', 'cluster', 'genre'])
if 'recommendation_message' not in st.session_state:
    st.session_state.recommendation_message = "Predict a cluster to get recommendations."


# --- Caching Model Loading ---
@st.cache_resource
def load_models():
    """Loads all models and frequency maps using caching."""
    scaler = None
    pca = None
    kmeans_optimal = None
    freq_maps = {}
    
    model_paths = {
        'scaler': 'models/scaler_model.pkl',
        'pca': 'models/pca_model.pkl',
        'kmeans_optimal': 'models/kmeans_model.pkl'
    }

    freq_map_paths = {
        'genre': 'models/genre_freq_map.pkl',
        'key': 'models/key_freq_map.pkl',
        'mode': 'models/mode_freq_map.pkl',
        'time_signature': 'models/time_signature_freq_map.pkl'
    }

    errors = []

    for model_name, path in model_paths.items():
        try:
            with open(path, 'rb') as file:
                if model_name == 'scaler':
                    scaler = pickle.load(file)
                elif model_name == 'pca':
                    pca = pickle.load(file)
                elif model_name == 'kmeans_optimal':
                    kmeans_optimal = pickle.load(file)
        except FileNotFoundError:
            errors.append(f"Model file '{path}' not found.")
        except Exception as e:
            errors.append(f"Error loading {model_name} model: {e}")

    for col_name, path in freq_map_paths.items():
        try:
            with open(path, 'rb') as file:
                freq_maps[col_name] = pickle.load(file)
        except FileNotFoundError:
            errors.append(f"Frequency map file '{path}' not found.")
        except Exception as e:
            errors.append(f"Error loading {col_name} frequency map: {e}")
            
    return scaler, pca, kmeans_optimal, freq_maps, errors

# --- Caching Recommendation Data Loading ---
@st.cache_data
def load_recommendation_data():
    """Loads the recommendation CSV data using caching."""
    rec_df = pd.DataFrame()
    error = None
    try:
        rec_df = pd.read_csv('Recommendation.csv')
        if 'cluster' in rec_df.columns:
            rec_df['cluster'] = rec_df['cluster'].astype(int)
    except FileNotFoundError:
        error = "Error: 'Recommendation.csv' not found. Please place the recommendation data file in the same directory as 'app.py'."
    except Exception as e:
        error = f"An error occurred while loading recommendation data: {e}"
    return rec_df, error

# Load all resources once at the start of the app
scaler, pca, kmeans_optimal, freq_maps, model_load_errors = load_models()
recommendation_df, rec_data_load_error = load_recommendation_data()

loaded_successfully = not (model_load_errors or rec_data_load_error)

if loaded_successfully:
    st.success("All resources loaded successfully!")
else:
    st.error("Some essential files could not be loaded. Please check the file paths and ensure all models and data are correctly placed.")
    for err in model_load_errors:
        st.error(err)
    if rec_data_load_error:
        st.error(rec_data_load_error)

st.write("---")

# --- Input Section ---
st.header("Enter Music Features")

col1, col2, col3 = st.columns(3)

numerical_feature_names = [
    'popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
    'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
]

categorical_feature_names = [
    'genre', 'key', 'mode', 'time_signature'
]

with col1:
    popularity = st.slider('Popularity (0-100)', 0, 100, 50)
    acousticness = st.number_input('Acousticness (0.0-1.0)', min_value=0.0000, max_value=1.0000, value=0.5, step=0.01)
    danceability = st.number_input('Danceability (0.0-1.0)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    duration_ms = st.number_input('Duration (ms)', min_value=0, max_value=9, value=1)
    energy = st.number_input('Energy (0.0-1.0)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

with col2:
    instrumentalness = st.number_input('Instrumentalness (0.0-1.0)', min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    liveness = st.number_input('Liveness (0.0-1.0)', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    loudness = st.number_input('Loudness (dB)', min_value=-60.0, max_value=0.0, value=-10.0, step=0.1)
    speechiness = st.number_input('Speechiness (0.0-1.0)', min_value=-100.0, max_value=15.0, value=-10.0, step=0.01)
    tempo = st.number_input('Tempo (BPM)', min_value=0.0, max_value=250.0, value=120.0, step=0.1)

with col3:
    valence = st.number_input('Valence (0.0-1.0)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
    genre_options = list(freq_maps.get('genre', {}).keys())
    key_options = list(freq_maps.get('key', {}).keys())
    mode_options = list(freq_maps.get('mode', {}).keys())
    time_signature_options = list(freq_maps.get('time_signature', {}).keys())

    genre_default_index = genre_options.index('Pop') if 'Pop' in genre_options else (0 if genre_options else None)
    key_default_index = key_options.index('C') if 'C' in key_options else (0 if key_options else None)
    mode_default_index = mode_options.index('Major') if 'Major' in mode_options else (0 if mode_options else None)
    time_signature_default_index = time_signature_options.index('4/4') if '4/4' in time_signature_options else (0 if time_signature_options else None)

    genre = st.selectbox('Genre', options=genre_options, index=genre_default_index, disabled=not genre_options, key='input_genre')
    key = st.selectbox('Key', options=key_options, index=key_default_index, disabled=not key_options, key='input_key')
    mode = st.selectbox('Mode', options=mode_options, index=mode_default_index, disabled=not mode_options, key='input_mode')
    time_signature = st.selectbox('Time Signature', options=time_signature_options, index=time_signature_default_index, disabled=not time_signature_options, key='input_time_signature')


all_pca_input_feature_names = numerical_feature_names + [
    'genre_encoded', 'key_encoded', 'mode_encoded', 'time_signature_encoded'
]

st.write("---")

def get_recommendations(rec_data_df, target_cluster, num_recs=3):
    if rec_data_df.empty:
        return pd.DataFrame(), "Recommendation data not loaded or is empty."

    filtered_songs = rec_data_df[rec_data_df['cluster'] == target_cluster]

    if filtered_songs.empty:
        return pd.DataFrame(), f"No songs found for Cluster {target_cluster} in the recommendation dataset."
    elif len(filtered_songs) < num_recs:
        return filtered_songs, f"Only {len(filtered_songs)} song(s) found for this cluster. Displaying all."
    else:
        return filtered_songs.sample(n=num_recs, random_state=random.randint(0, 1000000)), None


if st.button("Predict Music Cluster"):
    if loaded_successfully:
        try:
            numerical_input_data = pd.DataFrame([[
                popularity, acousticness, danceability, duration_ms, energy,
                instrumentalness, liveness, loudness, speechiness, tempo, valence
            ]], columns=numerical_feature_names)

            scaled_numerical_input = scaler.transform(numerical_input_data)
            scaled_numerical_df = pd.DataFrame(scaled_numerical_input, columns=numerical_feature_names)

            genre_encoded_val = freq_maps.get('genre', {}).get(genre, 0.0)
            key_encoded_val = freq_maps.get('key', {}).get(key, 0.0)
            mode_encoded_val = freq_maps.get('mode', {}).get(mode, 0.0)
            time_signature_encoded_val = freq_maps.get('time_signature', {}).get(time_signature, 0.0)

            encoded_categorical_df = pd.DataFrame([[
                genre_encoded_val, key_encoded_val, mode_encoded_val, time_signature_encoded_val
            ]], columns=['genre_encoded', 'key_encoded', 'mode_encoded', 'time_signature_encoded'])

            combined_features_for_pca = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)
            combined_features_for_pca = combined_features_for_pca[all_pca_input_feature_names]

            pca_transformed_input = pca.transform(combined_features_for_pca)
            pca_transformed_df = pd.DataFrame(pca_transformed_input, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

            predicted_cluster = kmeans_optimal.predict(pca_transformed_df)[0]

            st.success(f"The predicted music cluster for your input is: **Cluster {predicted_cluster}**")
            st.info("Songs in this cluster might share similar characteristics to the one you described.")

            st.session_state.predicted_cluster = predicted_cluster

            recs, msg = get_recommendations(
                recommendation_df,
                st.session_state.predicted_cluster
            )
            st.session_state.recommendations_data = recs
            st.session_state.recommendation_message = msg if msg else ""

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}.")
            st.error("Please ensure your models were trained following the exact pipeline: 1. Scale numerical features ONLY. 2. Frequency encode categorical features. 3. Concatenate (scaled numerical + frequency encoded categorical) in the correct order for PCA.")
            st.error("Also, ensure the loaded frequency maps contain the categories you expect.")
    else:
        st.warning("Cannot predict. Models and/or frequency maps failed to load. Please check the file paths.")

# --- Recommendation Display Section ---
st.write("---")
st.header("Recommended Songs")

if st.session_state.predicted_cluster is not None:
    st.subheader(f"Recommendations for Cluster {st.session_state.predicted_cluster}")
    
    if not st.session_state.recommendations_data.empty:
        for index, row in st.session_state.recommendations_data.iterrows():
            col_left, col_right = st.columns([0.7, 0.3])
            with col_left:
                st.write(f"🎶 **{row['track_name']}** by {row['artist_name']}")
            with col_right:
                st.markdown(f"<p style='text-align: right; font-size: 0.9em; color: grey;'>{row['genre']}</p>", unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info("No recommendations found for this cluster in the dataset.")

    if st.session_state.recommendation_message:
        st.warning(st.session_state.recommendation_message)

    if st.button("Refresh Recommendations 🔄"):
        recs, msg = get_recommendations(
            recommendation_df,
            st.session_state.predicted_cluster
        )
        st.session_state.recommendations_data = recs
        st.session_state.recommendation_message = msg if msg else ""
        st.rerun()
else:
    st.info("Predict a cluster first to see recommendations.")
