import streamlit as st
from pickle import load
import pandas as pd 
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

model = load(open('../models/knn_model.sav', 'rb'))
df_full = pd.read_csv('../data/raw/full_file.csv')

st.title('Movie Recommendation System')
st.write('Recommendations based on your favorite movie')
movie = st.text_input('Tell me which movie is your favorite')

vmodel = TfidfVectorizer()
vmatrix = vmodel.fit_transform(df_full['tags'])

model = NearestNeighbors(n_neighbors = 6, metric = 'cosine')
model.fit(vmatrix)

similarity = cosine_similarity(vmatrix)

def recommend(movie):
    movie_index = df_full[df_full["original_title"] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse = True , key = lambda x: x[1])[1:6]
    movies_names = []
    for movie in movie_list:
        movies_names.append(df_full.iloc[movie[0]].original_title)
    return movies_names
    
if st.button('Recommend'):
    recommendations = recommend(movie)
    for rec in recommendations:
        st.write(f'- {rec}')
    
    