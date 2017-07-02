#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All calculations necessary for the deployment of the application

Created on Thu Jun 15 19:11:43 2017

@author: chris
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os, requests
from bs4 import BeautifulSoup
import sqlite3

#%% Key variables:
Vt_loc = os.path.join('deploy','Vt_impbias_50.npy')
#Vt_loc = os.path.join('results','Vt_train_impbiasreg-3_50.npy')
valid_path = os.path.join('deploy','valid_anime.npy')
database = os.path.join('deploy','animeData.db')
N_anime = 12871

#%% Auxillary functions:
def get_cos_sim(threshold=0):
    ''' Calculate the cosine similarity '''
    Vt = np.load(Vt_loc)
    cos_sim = cosine_similarity(np.transpose(Vt))
    cos_sim[cos_sim<threshold]=0
    return cos_sim

def predict_rating(user_row,anime_idx,cos_sim_row,top_N,just_numerator=True):
    ''' Predict the rating of a single anime given a user'''
    cos = np.multiply(user_row!=0,cos_sim_row)
    if top_N:
        cos[np.argsort(cos)[:-(top_N-1)]]=0
    if just_numerator:
        return np.dot(user_row,np.transpose(cos))
    else:
        return np.dot(user_row,np.transpose(cos))/(10**-8+np.sum(cos))

def predict_top_ratings(user_row,cos_sim,top_N,N_recs):
    ''' Leverages the users ratings and the cosine similarities to predict
    a user's top rated anime. Current considers both the similarity to the 
    rated animes and the actual score to be equally important. '''
    # User data needs to be 1D:
    user_row=np.squeeze(user_row)
    # Find valid anime:
    valid_idxs = np.where(np.load(valid_path))[0]
    # Predict ratings on valid anime
    pratings=np.zeros_like(user_row)
    pratings[valid_idxs] = [predict_rating(user_row,x,cos_sim[x,:],top_N) for x in valid_idxs]
    # Remove anime the user has already rated
    rated = np.where(user_row)[0]
    recs = np.argsort(pratings)[::-1]
    recs = [x for x in recs if x not in rated]
    print([pratings[x] for x in recs[0:N_recs]])
    return recs[0:N_recs]

def get_user_data(user_id):
    ''' Get a user's anime info '''
    url = 'https://myanimelist.net/malappinfo.php?status=all&type=anime&u={0}'.format(user_id)
    page = requests.get(url)
    soup = BeautifulSoup(page.text,'lxml')
    # Extract all anime info:
    all_links = soup.find_all('anime')
    user_ratings = {}
    for link in all_links:
        anime_db_id = link.series_animedb_id.text
        user_ratings[int(anime_db_id)] = int(link.my_score.text)
    return  user_ratings

def convert_to_row(user_ratings,aid_dict,N_anime):
    ''' Converts user dictionary containing {anime_id:rating} to
    a row with scores in the correct locations '''
    # Convert anime ids to indices
    aid_dict_inv = {v:int(k) for k,v in aid_dict.items()}
    keys = [aid_dict_inv[x] if x in aid_dict_inv else None 
            for x in user_ratings.keys()]
    ratings = [x for key,x in zip(keys,list(user_ratings.values()))
                if key is not None]
    keys = [x for x in keys if x!=None]
    # Fill in the user data:
    user_row = np.zeros((1,N_anime))
    user_row[0,keys]=list(ratings)
    return user_row
    
def load_aid_dict(inverse=False):
    ''' Loads anime id to index dictionary '''
    con = sqlite3.connect(database)
    cur = con.cursor()
    if inverse:
        items = cur.execute('SELECT Anime,Aid FROM AnimeData').fetchall()
    else:
        items = cur.execute('SELECT Aid,Anime FROM AnimeData').fetchall()
    the_dict = {k:v for k,v in items}
    con.close()
    return the_dict

def get_top_predictions(user_row,top_N=6):
    ''' Returns info on the anime cooresponding to the top predictions '''
    aids = predict_top_ratings(user_row,get_cos_sim(),top_N=top_N,N_recs=6)
    # Request info from the database:
    con = sqlite3.connect(database)
    cur = con.cursor()
    recs = []
    for aid in aids:
        result = cur.execute('SELECT Anime, Image, Name, Score FROM AnimeData\
                             WHERE Aid={0}'.format(aid)).fetchone()
        recs.append(result)
    return recs

def get_names_ids():
    con = sqlite3.connect(database)
    cur = con.cursor()
    names_ids = cur.execute('SELECT Name,Anime from AnimeData ORDER BY Name').fetchall()
    con.close()
    return names_ids
#%% Main function:
def get_recs(user_id=None,user_ratings=None):
    if user_id:
        user_ratings = get_user_data(user_id)
    elif user_ratings:
        None
    else:
        return None
    user_row = convert_to_row(user_ratings,load_aid_dict(),N_anime)
    anime_ids = get_top_predictions(user_row,top_N=6)
    return anime_ids



