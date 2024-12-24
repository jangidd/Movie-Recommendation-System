import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies = pd.read_csv("movies_tags.csv")
def create_similarity(data):
    cv = CountVectorizer(max_features=5000,stop_words='english')
    vectors = cv.fit_transform(data.tags).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

def recommender(movie):
    similarity = create_similarity(movies)
    movie_index = movies[movies['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:11]
    x = pd.DataFrame()
    for i in movies_list:
        y = {'id':movies.iloc[i[0]].id,'movies':movies.iloc[i[0]].title}
        x = x.append(y,ignore_index=True)
    return x