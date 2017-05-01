# -*- coding: utf-8 -*-
"""
Collects user ids from club. Then, collects ratings from the user ids.

Created on Fri Apr 28 15:16:53 2017

@author: Chris Pierse
"""
import urllib2
from bs4 import BeautifulSoup
import numpy as np
import re, time, json
import sqlite3
#%% Gathering users through club data:
def crawl_club_users(cid):
    ''' Crawl through and get all user ids in a specific club'''
    club_users=[]
    page_num, count = 0, 36
    while count==36:
        # Prepare url:
        url = 'https://myanimelist.net/clubs.php?action=view&t=members&id={0}&show={1}'.format(cid,str(page_num*36))
        try:        
            page = urllib2.urlopen(url)
            soup = BeautifulSoup(page)
            # Extract all links:
            all_links = soup.find_all('a',href=True)
            count = 0
            for link in all_links:
                if 'profile' in link['href']:
                    if len(link.text)>0:
                        # These are the users
                        club_users.append(link.text)
                        count+=1
            # Get the next page number and rest:
            page_num +=1
            time.sleep(0.5+abs(np.random.randn(1)))
            if int(page_num)%10==9:
                print('Moving to page ' + str(page_num))
        except urllib2.HTTPError:
            count=0
    return club_users

def crawl_clubs(club_page_num_start=0, max_pages=20,club_sort='5'):
    ''' Crawl through and get club ids. club_sort = 5 gives top clubs '''
    club_ids = []
    club_page_num=club_page_num_start
    while club_page_num-club_page_num_start<max_pages:
        club_list_url = 'https://myanimelist.net/clubs.php?sort={0}&p={1}'.format(club_sort,str(club_page_num))
        page = urllib2.urlopen(club_list_url)
        soup = BeautifulSoup(page)
        all_links = soup.find_all('a',class_='fw-b',href=True)
        for link in all_links:
            # There should be one cid per link:
            cid = re.findall(r'=([0-9]+)',link['href'])[0]
            club_ids.append(cid)
        # Get the next page number and rest:
        club_page_num +=1
        time.sleep(1+abs(np.random.randn(1)))
        print('Moving to page ' + str(club_page_num))
    return club_ids

def get_users_from_clubs(club_ids,start_club=0,users=set()):
    ''' Crawl through and get the users of these clubs '''
    old_len = len(users)
    for idx,cid in enumerate(club_ids[start_club:]):
        print('Current cid: ' + str(cid))
        club_users = crawl_club_users(cid)
        users = set.union(users,set(club_users))
        if len(users)==old_len:
            print('Users did not change. Club_id: ' + str(cid))
        # Checkpoints
        if idx%2==1 or idx<10:
            print('SAVING CHECKPOINT: ' + str(idx))
            with open('user_ids_checkpoint.txt', 'w') as fp:
                for uid in users:
                  fp.write("{0}\n".format(uid))
                fp.close()
        old_len = len(users)
    return users

def get_user_info(user_id):
    ''' Get a user's anime info '''
    url = 'https://myanimelist.net/malappinfo.php?status=all&type=anime&u={0}'.format(user_id)
    page = urllib2.urlopen(url)
    soup = BeautifulSoup(page)
    # Extract all anime info:
    all_links = soup.find_all('anime')
    user_dict = {}
    anime_dict = {}
    for link in all_links:
        anime_db_id = link.series_animedb_id.text
        user_dict[anime_db_id] = {'score': link.my_score.text, 'status': link.my_status.text}
        anime_dict[anime_db_id] = {'name':link.series_title.text,'type':link.series_type.text,
                                  'synonyms': link.series_synonyms.text, 'image':link.series_image.text}
    time.sleep(0.1)
    return user_dict, anime_dict


def create_uid_dict(user_ids):
    # Load or create the uid_map
    try:
        with open('uid_map.json','r') as fp:    
            uid_dict = json.load(fp)
        uid = max(int(x) for x in uid_dict.keys())
        fresh = False
    except IOError:
        uid_dict = {}
        uid = 0
        fresh = True
    # Load the dict with new data:
    for user_id in user_ids:
        if fresh or user_id not in uid_dict.values():
            uid_dict[uid] = user_id
            uid+=1
    # Save the data:
    with open('uid_map.json','w') as fp:    
        json.dump(uid_dict, fp, sort_keys=True)
    return uid_dict


def setup_sql():
    con = sqlite3.connect('user_anime_data.db')
    cur = con.cursor()   
    cur.execute("CREATE TABLE IF NOT EXISTS \
        UserData(Uid INT, \
        Anime INT, \
        Score INT, \
        Status INT, \
        CONSTRAINT Uid_Anime PRIMARY KEY (Uid,Anime))")
        
    cur.execute("CREATE TABLE IF NOT EXISTS \
        AnimeData(Anime INT PRIMARY KEY, \
        Name VARCHAR, \
        Type INT, \
        Synonyms VARCHAR, \
        Image VARCHAR)")
    
    con.commit()
    return con, cur

def get_and_save_user_data(uid_dict,cur,con,cont=True):
    ''' Saves a user's info the the HDD '''
    if not cont:
        start = 0
    else:
        start = 1+cur.execute('SELECT MAX(Uid) FROM UserData').fetchone()[0]
    for uid in range(start,len(uid_dict)):
        user_id = uid_dict[str(uid)]
        user_dict, anime_dict = get_user_info(user_id)
        # Save user data to sql:
        try:
            cur.executemany("INSERT INTO UserData (Uid, Anime, Score, Status) VALUES (?,?,?,?)", 
                            [(uid, int(key), int(value['score']),int(value['status'])) 
                            for (key,value) in user_dict.items()])
        except sqlite3.IntegrityError:
            print('User ' + str(user_id) + ' found')
        # Save the anime data to sql:
        cur.executemany("INSERT OR IGNORE INTO AnimeData (Anime, Name, Type, Synonyms, Image) VALUES (?,?,?,?,?)", 
                        [(int(key), value['name'],int(value['type']),value['synonyms'],value['image'])
                        for (key,value) in anime_dict.items()] )
        if uid%25==24: 
            con.commit()
            print('Onto uid: ' + str(uid+1))
    




#%% Main code:

# Get and save Club Ids:
club_ids = crawl_clubs(club_page_num_start=0, max_pages=50,club_sort='5')
with open('club_ids.txt', 'w') as fp:
    for cid in club_ids:
      fp.write("{0}\n".format(cid))
    fp.close()

# Load club ids and get user ids
club_ids = open('club_ids.txt','r').read().split('\n')[:-1]
user_ids = get_users_from_clubs(club_ids)

# Continue this:
#user_ids = open('user_ids.txt','r').read().split('\n')[:-1]
#user_ids = get_users_from_clubs(club_ids,start_club=96,users=set(user_ids))
# Stopped at 727

# Map uids:
user_ids = open('user_ids.txt','r').read().split('\n')[:-1]
uid_dict = create_uid_dict(user_ids)

# Load uid_dict:
with open('uid_map.json','r') as fp:    
    uid_dict = json.load(fp)

# Start collecting user data:
con,cur = setup_sql()
get_and_save_user_data(uid_dict,cur,con,cont=True)
con.close()

