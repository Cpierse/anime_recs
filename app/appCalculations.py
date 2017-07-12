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
import json

#%% Key variables:
Vt_loc = os.path.join('deploy','Vt_impbias_50.npy')
#Vt_loc = os.path.join('deploy','Vt_impbiasreg-1.5_50.npy')
valid_path = os.path.join('deploy','valid_anime.npy')
database = os.path.join('deploy','animeData.db')
genre_dict_loc = os.path.join('deploy','genre_dict.json')
N_anime = 12871

#%% Auxillary functions:
def get_genre_dict():
    return json.load(open(genre_dict_loc,'r'))
    
def get_cos_sim(threshold=0):
    ''' Calculate the cosine similarity '''
    Vt = np.load(Vt_loc)
    cos_sim = np.array(cosine_similarity(np.transpose(Vt)),dtype=np.float32)
    cos_sim[cos_sim<threshold]=0
    return cos_sim

def predict_rating(user_row,anime_idx,cos_sim_row,top_N=20,thres=15):
    ''' Predict the rating of a single anime given a user'''
    cos = np.multiply(user_row!=0,cos_sim_row)
    if top_N:
        cos[np.argsort(cos)[:-(top_N-1)]]=0
    numerator = np.dot(user_row,np.transpose(cos))
    cos_sum = np.sum(cos)
    prating = numerator/(10**-8+cos_sum)
    return (prating*(numerator>thres), numerator)

def get_contributions(user_row,anime_idx,cos_sim_row,N_cont=3,cont_thresh=0):
    ''' Get the input anime that are responsible for this recommendation '''
    cos = np.multiply(user_row!=0,cos_sim_row)
    contributions = np.multiply(user_row,cos)
    contributions = sorted([(idx,x) for idx,x in enumerate(contributions) if x>0],key=lambda x:-x[1])
    return [x[0] for x in contributions[:N_cont] if x[1]>cont_thresh]

def predict_top_ratings(user_row,cos_sim,top_N,N_recs,related=None,genres=None):
    ''' Leverages the users ratings and the cosine similarities to predict
    a user's top rated anime. Current considers both the similarity to the 
    rated animes and the actual score to be equally important. '''
    # User data needs to be 1D:
    user_row=np.squeeze(user_row)
    # Find valid anime:
    print('Log: loading valid anime indices')
    valid_idxs = np.where(np.load(valid_path))[0]
    print('Log: Altering valid anime andices (if needed)')
    if related is not None:
        valid_idxs = [x for x in valid_idxs if x not in related]
    if genres is not None:
        in_genre = get_genre_anime(genres)
        valid_idxs = [x for x in valid_idxs if x in in_genre]
    # Predict ratings on valid anime
    print('Log: predicting...')
    pratings= [(x,predict_rating(user_row,x,cos_sim[x,:],top_N)) for x in valid_idxs]
    print('Log: predictions complete')
    # Remove anime the user has already rated
    rated = np.where(user_row)[0]
    recs = sorted(pratings, key = lambda x: (-x[1][0], -x[1][1]))
    recs = [x for x in recs if x[0] not in rated]
    # Filter related:
    print(recs[:N_recs])
    recs_plus = [(x[0],get_contributions(user_row,x[0],cos_sim[x[0],:])) for x in recs[0:N_recs]]
    return recs_plus

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

def get_top_predictions(user_row,top_N=20,related=None,genres=None):
    ''' Returns info on the anime cooresponding to the top predictions '''
    aids_plus = predict_top_ratings(user_row,get_cos_sim(),top_N=top_N,
                                    N_recs=6,related=related,genres=genres)
    # Request info from the database:
    con = sqlite3.connect(database)
    cur = con.cursor()
    recs = []
    for aid_plus in aids_plus:
        result = cur.execute('SELECT Anime, Image, Name, Score FROM AnimeData\
                             WHERE Aid={0}'.format(aid_plus[0])).fetchone()
        contributors = []
        for cont in aid_plus[1]:
            contributors.append(cur.execute('SELECT Image, Name FROM AnimeData\
                             WHERE Aid={0}'.format(cont)).fetchone())
        recs.append((result,contributors))
    return recs

def get_names_ids():
    con = sqlite3.connect(database)
    cur = con.cursor()
    names_ids = cur.execute('SELECT Name,Anime from AnimeData ORDER BY Name').fetchall()
    con.close()
    return names_ids

def get_related(anime_ids):
    con = sqlite3.connect(database)
    cur = con.cursor()
    related = cur.execute('SELECT Related_Aids from AnimeData WHERE Anime IN ({0})' \
                          .format(','.join([str(x) for x in anime_ids]))).fetchall()
    related = [int(x) for x in str.split(','.join([x[0] for x in related]),',')]
    con.close()
    return related

def get_genre_anime(genres):
    con = sqlite3.connect(database)
    cur = con.cursor()
    if len(genres)==1:
        in_genre = cur.execute('SELECT Aid from AnimeData WHERE Genres LIKE "%{0}%"' \
                      .format(genres[0])).fetchall()
        in_genre = [int(x[0]) for x in in_genre]
    if len(genres)>1:
        sql_base = 'OR Genres LIKE "%{0}%"'
        sql_end = ' '.join([sql_base.format(x) for x in genres[1:]])
        in_genre = cur.execute('SELECT Aid from AnimeData WHERE Genres LIKE "%{0}%" {1}' \
                      .format(genres[0],sql_end)).fetchall()
        in_genre = [int(x[0]) for x in in_genre]
    return in_genre


#%% Main functions:
def get_recs(user_id=None,user_ratings=None,related=None,genres=None):
    ''' Calculate a user's anime recommendations '''
    if user_id:
        print('Log: getting user data')
        user_ratings = get_user_data(user_id)
        print(user_ratings)
    elif user_ratings:
        None
    else:
        return None
    print('Log: getting aid_dict from db')
    aid_dict = load_aid_dict()
    if related:
        print('Log: getting related anime from db')
        related = get_related(list(user_ratings.keys()))
    print('Log: aid_dict acquired. Converting user row')
    user_row = convert_to_row(user_ratings,aid_dict,N_anime)
    print('Log: converted user data to row')
    anime_ids = get_top_predictions(user_row,top_N=20,related=related,genres=genres)
    return anime_ids

def get_results(recs_plus):
    ''' Process calculated recs to get results for html '''
    recs = [x[0] for x in recs_plus]
    images = ['https://myanimelist.cdn-dena.com/images/anime/{0}'
              .format(x[1]) for x in recs]
    urls = ['https://myanimelist.net/anime/{0}'
              .format(x[0]) for x in recs]
    names = [x[2] for x in recs]
    # Handle anime that contributed to these recs:
    conts = [x[1] for x in recs_plus]
    conts = [[('https://myanimelist.cdn-dena.com/images/anime/{0}'
              .format(x[0]),x[1]) for x in group] for group in conts]
    # Return all results including an index
    indices = [x for x in range(0,len(images))]
    return zip(indices,images,urls,names,conts)

