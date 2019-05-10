"""
Author : Loris Wintjens
Goal : Practicing and showing my skills using ML
Title : Movie recommendation system
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# helper functions
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

# Read the data set from the CSV file
df = pd.read_csv("movie_dataset.csv")

# Select features 
# Watch out, if you use any other features, you might need to reprocess it before using it, I didn't reprocess other features than the ones implemented in this script
features = ['keywords','cast','genres','director']

# Create a column in DF which combines all selected features
for feature in features :
    # we avoid NaN values
    df[feature] = df[feature].fillna('')

def combineFeatures(row):
    # combine all the features in a row
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']

df["combined_features"] = df.apply(combineFeatures, axis = 1)

# Create count matrix from this new combined column
cv = CountVectorizer()
countMatrix = cv.fit_transform(df["combined_features"])

# Compute the cosine similarity based on the countMatrix
similarity_scores = cosine_similarity(countMatrix)

# Avatar will be the movie that the user likes, based on this movie we will recommend similar movies
movie_user_likes = "Avatar"

# Get index of this movie from its title using the helper function get_index_from_title(title)
movieIndex = get_index_from_title(movie_user_likes)

# Get a sorted list of similar movies depending on their similarity score
similarMovies = list(enumerate(similarity_scores[movieIndex])) 
# similarMovies is a list of tuples : (index, similarity_score)

# we sort the list in a descending order of similarity score
sortedSimilarMovies = sorted(similarMovies,key= lambda x: x[1], reverse= True)

# Print the titles of the first 50 movies
for (counter,movie) in enumerate (sortedSimilarMovies):
    if counter == 0 :
        print('We are looking similar movies to: ', get_title_from_index(movie[0]))
        print('We print the 50 first similar movies, from the most similar to the less one.')
        print('============================================================================')
    elif counter <= 50 :
        print(counter, ' ', get_title_from_index(movie[0]))
