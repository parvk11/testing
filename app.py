from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib

app = FastAPI()

origins = ["*"]
app.add_middleware(
 CORSMiddleware,
 allow_origins=origins,
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)

@app.get("/api/test")
async def test():
 return "Hello World!"

model = joblib.load("movie-recommender.pkl")

class UserInput(BaseModel):
    movie_w_rating: dict

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


X = model["X"]
Y = model["Y"]
b = model["b"]
W = model["W"]
df = model["ratings_df"]
df2 = model["movie_df"]
average_ratings = model["average_ratings"]
all_movies = model['all_movies']
num_features = X.shape[1]
num_movies = X.shape[0]
user_w = tf.Variable(tf.random.normal((1,  num_features),dtype=tf.float64),  name='user_w')
optimizer = keras.optimizers.Adam(learning_rate=.1)

def show_top(topindices, predicted_user_movies, df2,rated_movie_ids):
    newdf2 = df2.copy()
    ratings_count = df.groupby('movieId').size().reset_index(name='count')
    movie_ids = newdf2['movieId'].values
    ratings_count_movieids = ratings_count["movieId"].values
    dropped = [] ##drop movies that have never been rated
    for id in movie_ids:
        if(id not in ratings_count_movieids):
            index = newdf2[newdf2['movieId'] == id].index
            newdf2 = newdf2.drop(index)
            dropped.append(index)
    
    
    newdf2["count"] = ratings_count['count'].values
    newdf2["My Prediction"] = predicted_user_movies

    new_topindices = [x for x in topindices if x not in dropped]

    top_movies = newdf2.loc[new_topindices[:4500]]
    filter = top_movies["count"] > 30
    top_movies = top_movies[filter]

    for id in rated_movie_ids:
        index = top_movies[top_movies['movieId'] == id].index
        top_movies = top_movies.drop(index)

    return top_movies


def predict_ratings_user(X,W,average_ratings):
    predicted = (tf.linalg.matmul(X,tf.transpose(W)))
    predicted= predicted.numpy()
    for i in range(tf.shape(predicted)[0]):
        predicted[i] += average_ratings.values[i]
    clipped_predictions = tf.clip_by_value(predicted, 0.0, 5.0)
    return clipped_predictions

def get_movie_id(movie_title):
    movie = df2.loc[df2['title'] == movie_title]
    if not movie.empty:
        return movie['movieId'].index.tolist()[0]
    else:
        return None
def get_movie_name(index):
    if index < 0 or index >= len(df2):
        return "Index out of bounds"
    return df2.iloc[index]['title']
def get_real_movieid(movie_title):
    # pattern = 'movie_title'
    # movie = df2[df2['title'].str.contains(pattern, case=False, na=False)]
    movie = df2.loc[df2['title'] == movie_title]
    if not movie.empty:
        return movie['movieId'].tolist()
    else:
        return None
    
def make_user_ratings(user_dict):
    movie_dict = {movie:0 for movie in all_movies}
    for key,value in user_dict.items():
        movie_dict = update_ratings(key, value, movie_dict, all_movies)
    users_ratings =np.array(list(movie_dict.values()))

    rated_indices = []
    rated_movies_names = []
    rated_movie_ids = []
    count = 0
    for key in movie_dict.keys():
        if(movie_dict[key] != 0):
            rated_movie_ids.append(key)
            rated_indices.append(count)
            index = df2.index[df2['movieId'] == key].tolist()[0]
            rated_movies_names.append(get_movie_name(index))
        count += 1

    print(rated_movies_names)
    print(rated_indices)
    print(rated_movie_ids)
    return rated_movies_names, rated_indices, rated_movie_ids, users_ratings


def update_ratings(movie_name,rating,movie_ratings, all_movies):
    id = (get_movie_id(movie_name))
    if(id == None):
        print(f"{movie_name} not found")
        return movie_ratings
    
    movie = (get_movie_name(id))
    if(movie == None):
        print(f"{movie_name} not found")
        return movie_ratings
    key = (get_real_movieid(movie)[0])
    movie_ratings[key] = rating
    return movie_ratings

@tf.function
def get_user_recommendations(user_ratings, X, b, lam, iterations, user_w):
    
    user_r = [0]*num_movies
    user_ratings_tensor = tf.convert_to_tensor(user_ratings, dtype=tf.float64)
    user_r = tf.cast(user_ratings_tensor > 0, dtype=tf.float64)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            error = ((tf.matmul(X,tf.transpose(user_w)))- tf.reshape(user_ratings, (-1,1))) * tf.reshape(user_r, (-1,1))
            cost = (tf.reduce_sum(tf.square(error)) + (lam/2) *  tf.reduce_sum(tf.square(user_w)))
        gradients = tape.gradient(cost, [user_w])
        optimizer.apply_gradients(zip(gradients, [user_w]))
        if(i % 10 == 0):
            print(f'Cost at epoch {i}: {cost}')
    return user_w


@app.post("/api/showmovies")
async def recommend(ratings : dict):
    ratings_from_user = ratings


    rm_names, rm_inds, rm_mids, us_ratings = make_user_ratings(ratings_from_user)
    my_ratings_normal = us_ratings - average_ratings


    my_weights = get_user_recommendations(my_ratings_normal, X, b, lam = 1, iterations = 20,user_w = user_w)


    my_new_predicted = predict_ratings_user(X,my_weights,average_ratings)


    my_new_predicted = tf.squeeze(my_new_predicted)
    newix = tf.argsort(my_new_predicted, direction = "DESCENDING")
    best = show_top(topindices= newix, predicted_user_movies= my_new_predicted, df2 = df2, rated_movie_ids = rm_mids)
    best = best.sort_values("My Prediction", ascending = False)
    keys = best["title"].tolist()

    
    # values = best["My Prediction"].tolist()
    # new_dict = {}
    # for key,value in zip(keys, values):
    #     new_dict[key] = value
    # return new_dict
    return JSONResponse(content = best.to_json(orient = 'split'))