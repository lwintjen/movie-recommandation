from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
text = ["London Paris London", "Paris Paris London"]

cv = CountVectorizer()

countMatrix = cv.fit_transform(text)
   
similarity_scores = cosine_similarity(countMatrix)

print(similarity_scores)