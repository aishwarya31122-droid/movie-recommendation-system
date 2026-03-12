import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Merge data
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Create pivot table
movie_matrix = movie_ratings.pivot_table(
    index='title',
    columns='userId',
    values='rating'
).fillna(0)

# Train model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(movie_matrix)

# Recommendation function
def recommend(movie_name):
    movie_index = movie_matrix.index.get_loc(movie_name)

    distances, indices = model.kneighbors(
        movie_matrix.iloc[movie_index,:].values.reshape(1,-1),
        n_neighbors=6
    )

    recommended_movies = []
    for i in range(1,len(indices.flatten())):
        recommended_movies.append(movie_matrix.index[indices.flatten()[i]])

    return recommended_movies


# Streamlit UI
st.title("Movie Recommendation System 🎬")

movie_list = movie_matrix.index.tolist()

selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend Movies"):
    recommendations = recommend(selected_movie)

    st.write("Recommended Movies:")
    for movie in recommendations:
        st.write(movie)
        st.write(movies.head())