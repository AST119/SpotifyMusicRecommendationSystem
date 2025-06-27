Spotify Music Recommendation System using KMeans Clustering
Overview
This project is a Spotify Music Recommendation System that uses KMeans Clustering to suggest songs based on user preferences. The system analyzes audio features of songs and groups them into clusters, allowing users to discover new tracks similar to their favorite ones.

The application is built using:

Python for backend logic

Streamlit for the interactive web interface

Scikit-learn for KMeans clustering

Spotify API to fetch song data and audio features

🔗 Live Demo: https://spotifymusicrecommendationsystem.onrender.com

Features
Search for any song and get recommendations

Cluster-based recommendations using KMeans

Audio feature analysis (danceability, energy, valence, etc.)

Responsive UI with an intuitive design

How It Works
Input a Song: Enter a song name and artist to fetch its audio features.

Clustering: The system uses a pre-trained KMeans model to assign the song to a cluster.

Recommendations: Suggests similar songs from the same cluster.

Usage
Visit the live demo.

Enter a song name and artist.

Click "Get Recommendations" to see similar tracks.

Requirements
To run locally:

bash
pip install streamlit scikit-learn pandas spotipy  
Note
The app may take a few seconds to load on Render due to the free-tier server spin-up time.

Enjoy discovering new music! 🎵
