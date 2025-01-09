from fastapi import FastAPI, Depends, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
import numpy as np

from util import make_user_ratings, get_user_recommendations, predict_ratings_user, show_top, get_password_hash, get_current_user, verify_password, create_access_token, User, UserPreference, get_db, UserToWatch
from util import X, b, average_ratings, user_w, df2
from sqlalchemy import Column, Integer, String, create_engine, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
import logging
import time


SECRET_KEY = '6e8d5097c32ef72dba8a1f17aac1f6da261e489b34432ce4cbbbb85c46be107aa4847ae9317d74e1fa327473fdaf769b00b60d034e48c641c1ff3f0de4f890c7cf35934e8919beefd5106d007dd4076fd85d097e52397c3dd62710fbdb33f1efb5874a98d20be049bd6eb7559ff4b59abc784e50303f537016878dd4c6d90cb3'
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


logging.basicConfig(level=logging.INFO)


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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log the request details
    logging.info(f"Incoming request: {request.method} {request.url}")
    logging.info(f"Headers: {request.headers}")
    body = await request.body()
    logging.info(f"Body: {body.decode('utf-8')}")

    # Process the request
    response = await call_next(request)

    # Log the response details
    process_time = time.time() - start_time
    logging.info(f"Response status: {response.status_code}")
    logging.info(f"Process time: {process_time:.2f} seconds")

    return response

@app.get("/api/movies")
def get_movies():
    movies = df2['title'].tolist()
    # print(movies)
    return movies
@app.post("/api/register")
def register(username: str = Query(...), password: str = Query(...), db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed_password = get_password_hash(password)
    new_user = User(username=username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}

@app.post("/api/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    print("hello")
    user = db.query(User).filter(User.username == form_data.username).first()
    print(user)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": str(user.id)}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@app.delete("/api/delete")
def delete_user(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.delete(current_user)
    db.query(UserPreference).filter(UserPreference.user_id == current_user.id).delete()
    db.commit()
    return {"message": "User deleted successfully"}

#add movie to watchlist
@app.post("/api/watchlist/{movie}")
async def add_movie(movie: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # check if the movie already exists in the user's watchlist
    existing_preference = db.query(UserToWatch).filter(
        UserToWatch.user_id == current_user.id,
        UserToWatch.movie_title == movie
    ).first()
    
    if existing_preference:
        raise HTTPException(status_code=400, detail="Movie already exists in watchlist")
    else:
        new_preference = UserToWatch(user_id=current_user.id, movie_title=movie)
        db.add(new_preference)
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=400, detail="Error saving user preferences") 
        return {"message": "Movie added successfully"}
@app.get("/api/watchlist")
def get_watchlist(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user_watchlist = db.query(UserToWatch).filter(UserToWatch.user_id == current_user.id).all()
    watchlist = [preference.movie_title for preference in user_watchlist]
    return watchlist
@app.delete("/api/watchlist/{movie}")
def delete_movie_from_watchlist(movie: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(UserToWatch).filter(UserToWatch.user_id == current_user.id, UserToWatch.movie_title == movie).delete()
    db.commit()
    return {"message": "Movie deleted successfully"}


@app.post("/api/showmovies")
async def recommend(ratings : dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    #process user ratings
    ratings_from_user = ratings
    rm_names, rm_inds, rm_mids, us_ratings = make_user_ratings(ratings_from_user)
    my_ratings_normal = us_ratings - average_ratings

    #get recommendations
    my_weights = get_user_recommendations(my_ratings_normal, X, b, lam = 1, iterations = 20,user_w = user_w)
    my_new_predicted = predict_ratings_user(X,my_weights,average_ratings)

    #get top recommendations
    my_new_predicted = tf.squeeze(my_new_predicted)
    newix = tf.argsort(my_new_predicted, direction = "DESCENDING")
    best = show_top(topindices= newix, predicted_user_movies= my_new_predicted, df2 = df2, rated_movie_ids = rm_mids)
    best = best.sort_values("My Prediction", ascending = False)
    keys = best["title"].tolist()
    genres = best["genres"].tolist()
    rating_user = best["My Prediction"].tolist()
    keys_genres = []
    for i in range(len(keys)):
        keys_genres.append([keys[i],genres[i],rating_user[i]])




    # Save ratings while avoiding duplicates
    for movie, rating in ratings.items():
        # Check if the movie already exists for the current user
        existing_preference = db.query(UserPreference).filter(
            UserPreference.user_id == current_user.id,
            UserPreference.movie_title == movie
        ).first()

        if existing_preference:
            # If the movie already exists, update the rating
            existing_preference.rating = rating
        else:
            # If the movie does not exist, create a new entry
            new_preference = UserPreference(user_id=current_user.id, movie_title=movie, rating=rating)
            db.add(new_preference)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="Error saving user preferences")

    # return JSONResponse(content = best.to_json(orient = 'split'))
    print(keys_genres[:100])
    return keys_genres[:100]

@app.get("/api/userprefernces")
def get_user_preferences(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user_preferences = db.query(UserPreference).filter(UserPreference.user_id == current_user.id).all()
    preferences_list = [
        {"movie_title": preference.movie_title, "rating": preference.rating}
        for preference in user_preferences
    ]

    return preferences_list
@app.delete("/api/userprefernces/{movie}")
def delete_movie(movie: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(UserPreference).filter(UserPreference.user_id == current_user.id, UserPreference.movie_title == movie).delete()
    db.commit()
    return {"message": "Movie deleted successfully"}