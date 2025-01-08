from fastapi import FastAPI, Depends, HTTPException, status
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
import joblib
from sqlalchemy import Column, Integer, String, create_engine, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os

tf.config.run_functions_eagerly(True)


SECRET_KEY = '6e8d5097c32ef72dba8a1f17aac1f6da261e489b34432ce4cbbbb85c46be107aa4847ae9317d74e1fa327473fdaf769b00b60d034e48c641c1ff3f0de4f890c7cf35934e8919beefd5106d007dd4076fd85d097e52397c3dd62710fbdb33f1efb5874a98d20be049bd6eb7559ff4b59abc784e50303f537016878dd4c6d90cb3'
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


model = joblib.load("movie-recommender.pkl")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Database setup
DATABASE_URL = "sqlite:///./movie_recommender.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_title = Column(String, index=True, nullable=False)
    rating = Column(Integer, nullable=False)
class UserToWatch(Base):
    __tablename__ = "user_to_watch"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_title = Column(String, index=True, nullable=False)

Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def  get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = db.query(User).filter(User.id == int(user_id)).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

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
    print(user_r, user_ratings_tensor)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            error = ((tf.matmul(X,tf.transpose(user_w)))- tf.reshape(user_ratings, (-1,1))) * tf.reshape(user_r, (-1,1))
            cost = (tf.reduce_sum(tf.square(error)) + (lam/2) *  tf.reduce_sum(tf.square(user_w)))  
        gradients = tape.gradient(cost, [user_w])
        optimizer.apply_gradients(zip(gradients, [user_w]))
        if(i % 10 == 0):
            print(f'Cost at epoch {i}: {cost.numpy()}')
    return user_w