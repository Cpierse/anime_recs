#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:53:38 2017

@author: chris
"""
import numpy as np
import sqlite3, os
from processData import get_aid_dict
from getData import setup_sql

#%% Key Variables:
deploy_database_loc = os.path.join('app','deploy','animeData.db')
valid_anime_loc = os.path.join('app','deploy','valid_anime.npy')

#%% Key Functions:

def deploy_database():
    # Load the existing data
    con, cur = setup_sql()
    data = cur.execute('SELECT Anime,Image,English_Name,Name,Score FROM \
                       AnimeData').fetchall()
    con.close()
    aid_dict = get_aid_dict()
    aid_dict_inv = {int(v):int(k) for k,v in aid_dict.items()}
    valid_anime = np.load(valid_anime_loc)
    
    # Create the new database:
    con = sqlite3.connect(deploy_database_loc)
    cur = con.cursor()
    cur.execute("CREATE TABLE \
        AnimeData(Anime INT PRIMARY KEY, \
        Aid INT, \
        Image VARCHAR,\
        Name VARCHAR, \
        Score FLOAT32 )")
    
    # Logic to fill new database:
    for row in data:
        anime_id = row[0]
        try: aid = aid_dict_inv[anime_id]
        except KeyError: print("{0} not found!".format(anime_id)); continue
        if row[2] and len(row[2])>0: name = row[2].replace('"',"'")
        else: name = row[3].replace('"',"'")
        image = '/'.join(str.split(row[1],'/')[-2:])
        score = row[4]
        if valid_anime[aid]:
                cur.execute('INSERT INTO AnimeData \
                            (Anime, Aid, Image, Name,Score) \
                            VALUES ({0},{1},"{2}","{3}",{4})'\
                            .format(anime_id,aid,image,name,score))
    con.commit()
    con.close()
        
        
        
    
    
