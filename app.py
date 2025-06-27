import streamlit as st
import pandas as pd
import pickle
import os

# Set page config for a cleaner look
st.set_page_config(layout="centered", page_title="Spotify Recommendation")

# --- Heading and Image ---
st.title("Spotify Music Recommendation System")

try:
    st.image("image.jpg", caption="Discover new music clusters!")
except FileNotFoundError:
    st.warning("image.jpg not found. Please place the image file in the same directory as the script.")

st.write("---")

# --- Load Models and Frequency Maps ---
scaler = None
pca = None
kmeans_optimal = None
freq_maps = {}

model_paths = {
    'scaler': 'models\scaler_model.pkl',
    'pca': 'models\pca_model.pkl',
    'kmeans_optimal': 'models\kmeans_model.pkl'
}

freq_map_paths = {
    'genre': 'models\genre_freq_map.pkl',
    'key': 'models\key_freq_map.pkl',
    'mode': 'models\mode_freq_map.pkl',
    'time_signature': 'models\time_signature_freq_map.pkl'
}

loaded_successfully = True

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
        st.error(f"Error: Model file '{path}' not found. Please ensure all model files are in the same directory.")
        loaded_successfully = False
    except Exception as e:
        st.error(f"An error occurred while loading {model_name} model: {e}")
        loaded_successfully = False

for col_name, path in freq_map_paths.items():
    try:
        with open(path, 'rb') as file:
            freq_maps[col_name] = pickle.load(file)
    except FileNotFoundError:
        st.error(f"Error: Frequency map file '{path}' not found. Please ensure all frequency map files are in the same directory.")
        loaded_successfully = False
    except Exception as e:
        st.error(f"An error occurred while loading {col_name} frequency map: {e}")
        loaded_successfully = False

if loaded_successfully:
    st.success("All models and frequency maps loaded successfully!")
else:
    st.error("Some essential files could not be loaded. Please check the file paths.")

st.write("---")

# --- Input Section ---
st.header("Enter Music Features")

col1, col2, col3 = st.columns(3)

# Define numerical feature names (those that were scaled)
numerical_feature_names = [
    'popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
    'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
]

# Define categorical feature names (those that were frequency encoded)
categorical_feature_names = [
    'genre', 'key', 'mode', 'time_signature'
]

with col1:
    popularity = st.slider('Popularity (0-100)', 0, 100, 50)
    acousticness = st.number_input('Acousticness (0.0-1.0)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    danceability = st.number_input('Danceability (0.0-1.0)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    duration_ms = st.number_input('Duration (ms)', min_value=0, max_value=600000, value=200000)
    energy = st.number_input('Energy (0.0-1.0)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

with col2:
    instrumentalness = st.number_input('Instrumentalness (0.0-1.0)', min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    liveness = st.number_input('Liveness (0.0-1.0)', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    loudness = st.number_input('Loudness (dB)', min_value=-60.0, max_value=0.0, value=-10.0, step=0.1)
    speechiness = st.number_input('Speechiness (0.0-1.0)', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
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

    genre = st.selectbox('Genre', options=genre_options, index=genre_default_index, disabled=not genre_options)
    key = st.selectbox('Key', options=key_options, index=key_default_index, disabled=not key_options)
    mode = st.selectbox('Mode', options=mode_options, index=mode_default_index, disabled=not mode_options)
    time_signature = st.selectbox('Time Signature', options=time_signature_options, index=time_signature_default_index, disabled=not time_signature_options)


# Define the final feature names that went into PCA in their exact order
# This order is crucial and must match how they were concatenated before PCA in training
all_pca_input_feature_names = numerical_feature_names + [
    'genre_encoded', 'key_encoded', 'mode_encoded', 'time_signature_encoded'
]

st.write("---")

if st.button("Predict Music Cluster"):
    if loaded_successfully:
        try:
            # 1. Prepare numerical features for scaling
            numerical_input_data = pd.DataFrame([[
                popularity, acousticness, danceability, duration_ms, energy,
                instrumentalness, liveness, loudness, speechiness, tempo, valence
            ]], columns=numerical_feature_names)

            # 2. Scale numerical features
            scaled_numerical_input = scaler.transform(numerical_input_data)
            scaled_numerical_df = pd.DataFrame(scaled_numerical_input, columns=numerical_feature_names)

            # 3. Apply Frequency Encoding to categorical inputs
            genre_encoded_val = freq_maps.get('genre', {}).get(genre, 0.0)
            key_encoded_val = freq_maps.get('key', {}).get(key, 0.0)
            mode_encoded_val = freq_maps.get('mode', {}).get(mode, 0.0)
            time_signature_encoded_val = freq_maps.get('time_signature', {}).get(time_signature, 0.0)

            encoded_categorical_df = pd.DataFrame([[
                genre_encoded_val, key_encoded_val, mode_encoded_val, time_signature_encoded_val
            ]], columns=['genre_encoded', 'key_encoded', 'mode_encoded', 'time_signature_encoded'])

            # 4. Combine scaled numerical features and encoded categorical features
            # Ensure the order of concatenation matches how PCA was trained
            combined_features_for_pca = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)

            # Crucial check: Ensure the column names and order match exactly
            combined_features_for_pca = combined_features_for_pca[all_pca_input_feature_names]

            # 5. Apply PCA transformation
            pca_transformed_input = pca.transform(combined_features_for_pca)
            pca_transformed_df = pd.DataFrame(pca_transformed_input, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

            # 6. Predict the cluster
            predicted_cluster = kmeans_optimal.predict(pca_transformed_df)[0]

            st.success(f"The predicted music cluster for your input is: **Cluster {predicted_cluster}**")
            st.info("Songs in this cluster might share similar characteristics to the one you described.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}.")
            st.error("Please ensure your models were trained following the exact pipeline:")
            st.error("1. Scale numerical features ONLY.")
            st.error("2. Frequency encode categorical features.")
            st.error("3. Concatenate (scaled numerical + frequency encoded categorical) in the correct order for PCA.")
            st.error("Also, ensure the loaded frequency maps contain the categories you expect.")
    else:
        st.warning("Cannot predict. Models and/or frequency maps failed to load. Please check the file paths.")