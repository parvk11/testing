from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib
from util import make_user_ratings, get_user_recommendations, predict_ratings_user, show_top
from util import X, b, average_ratings, user_w, df2

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