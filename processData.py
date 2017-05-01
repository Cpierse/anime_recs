# -*- coding: utf-8 -*-
"""
Analyzes the distributions of anime ratings. Seeks adjusted cosine similarity
between anime. 

Created on Fri Apr 28 17:12:02 2017

@author: Chris Pierse
"""
import numpy as np
import json, sqlite3, scipy, requests
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import lines
from PIL import Image
from StringIO import StringIO
import time

database = 'bu//user_anime_data.db'
status_dict = {'1':'Watching', '2':'Completed', '3':'On-Hold', '4':'Dropped', '5':'????', '6':'Plan to Watch'}
plt.style.use('ggplot')

#%% Key functions:
def get_aid_dict(load = True):
    ''' Update anime dict '''
    if load:
        with open('aid_map.json','r') as fp:    
            aid_dict = json.load(fp)
        aid_dict = {int(k):int(v) for k,v in aid_dict.items()}
    else:
        con = sqlite3.connect(database)
        cur = con.cursor()
        all_anime = cur.execute('SELECT Anime FROM AnimeData').fetchall()
        all_anime = [x[0] for x in all_anime]
        aid_dict=dict(zip(range(len(all_anime)),all_anime))
        with open('aid_map.json','w') as fp:    
            json.dump(aid_dict, fp, sort_keys=True)
        con.close()
    aid_dict_inv = {v: k for k, v in aid_dict.iteritems()}
    return aid_dict, aid_dict_inv

def process_user_ratings(aid_dict_inv, user_mean_centered = True):
    ''' Convert user data to jsons '''
    con = sqlite3.connect(database)
    cur = con.cursor()
    max_uid = cur.execute('SELECT MAX(Uid) FROM UserData').fetchone()[0]
    try:
        if user_mean_centered:
            data = scipy.sparse.lil_matrix((max_uid,len(aid_dict)),dtype=scipy.float32)
            for uid in range(0,max_uid):
                user_data = cur.execute('SELECT Anime, Score FROM UserData Where Uid=={0}'.format(uid)).fetchall()
                user_data = np.array([[x[0],x[1]] for x in user_data if x[1]>0])
                if  user_data.shape[0]>0:
                    mean_rating = np.mean(user_data[:,1])
                    data[uid,[aid_dict_inv[x[0]] for x in user_data]]=[scipy.float32(x[1]-mean_rating) for x in user_data]
        else:
            data = scipy.sparse.lil_matrix((max_uid,len(aid_dict)),dtype=scipy.int8)
            for uid in range(0,max_uid):
                user_data = cur.execute('SELECT Anime, Score FROM UserData Where Uid=={0}'.format(uid)).fetchall()
                data[uid,[aid_dict_inv[x[0]] for x in user_data]]=[x[1] for x in user_data]
    except KeyError:
        print('Update your anime dict')
        raise
    con.close()
    return data

def shorten_numbers(x, pos):
    ''' Clean up the plots by removing zeros'''
    if x >= 1e6:
        return '{:1.1f}M'.format(x*1e-6)
    if x==0:
        return '0'
    return '{:1.0f}K'.format(x*1e-3)

def rating_distributions():
    ''' Display histogram of rating counts '''
    con = sqlite3.connect(database)
    cur = con.cursor()
    user_summary = cur.execute('Select Uid, Count(Uid),Sum(Score) FROM UserData \
                                 WHERE Score>0 GROUP BY Uid ').fetchall()
    user_summary = np.array([[x[0], x[1], x[2], float(x[2])/x[1]]  for x in user_summary])
    all_ratings = [x[0] for x in  cur.execute('Select Score FROM \
                                         UserData  WHERE Score>0 ').fetchall()]
    # Determine whether those who rate more anime have a different average rating.
    bin_size, avg_ratings = 50, []
    num_ratings = np.arange(bin_size/2,1000,bin_size)
    for mid_bin in num_ratings:
        rel_data = user_summary[np.multiply(user_summary[:,1]>=mid_bin-bin_size/2,user_summary[:,1]<mid_bin+bin_size/2),1:3]
        avg_ratings.append( float(sum(rel_data[:,1]))/float(sum(rel_data[:,0])))
        #avg_ratings.append(np.mean(user_summary[np.multiply(user_summary[:,1]>=mid_bin-bin_size/2,user_summary[:,1]<mid_bin+bin_size/2),3]))
    # Plot these exploratory figures:
    f, axarr = plt.subplots(2,2)
    axarr[0,0].set_xlabel('Number of ratings (per user)',size=10)
    axarr[0,0].set_ylabel('Number of users',size=10)
    #axarr[0,0].text(-0.3, 1.05, 'A.', transform=axarr[0,0].transAxes, size=10)
    axarr[0,0].hist(user_summary[:,1],bins=np.arange(0,np.max(user_summary[:,1])+50,50))
    axarr[0,0].set_xlim([0,1000])
    axarr[0,0].yaxis.set_major_formatter(FuncFormatter(shorten_numbers))
    axarr[1,0].set_xlabel('Number of ratings (per user)',size=10)
    axarr[1,0].set_ylabel('Average rating',size=10)
    axarr[1,0].plot(num_ratings,avg_ratings)
    axarr[1,0].set_xlim([0,1000])
    axarr[1,0].set_yticks(range(0,11))
    axarr[1,0].set_yticklabels(['0','','2','','4','','6','','8','','10'])
    axarr[1,0].set_ylim([1,10])
    #axarr[1,0].text(-0.3, 1.05, 'B.', transform=axarr[1,0].transAxes, size=10)
    axarr[0,1].set_xlabel('Ratings',size=10)
    axarr[0,1].set_ylabel('Number of ratings ',size=10)
    axarr[0,1].hist(all_ratings,bins=np.arange(0.5,11.5))
    axarr[0,1].set_xlim([0.5,10.5])
    axarr[0,1].yaxis.set_major_formatter(FuncFormatter(shorten_numbers))
    #axarr[0,1].text(-0.3, 1.05, 'C.', transform=axarr[0,1].transAxes, size=10)
    plt.sca(axarr[0, 1])
    plt.xticks(range(1,11))
    axarr[1,1].set_xlabel('Average ratings (per user)',size=10)
    axarr[1,1].set_ylabel('Number of users',size=10)
    axarr[1,1].hist(user_summary[:,3],bins=np.arange(0.5,11.5))
    axarr[1,1].set_xlim([0.5,10.5])
    axarr[1,1].yaxis.set_major_formatter(FuncFormatter(shorten_numbers))
    #axarr[1,1].text(-0.3, 1.05, 'D.', transform=axarr[1,1].transAxes, size=10)
    plt.sca(axarr[1, 1])
    plt.xticks(range(1,11))
    f.tight_layout()
    for (i,j) in ((0,0),(0,1),(1,0),(1,1)):
            axarr[i,j].xaxis.set_ticks_position('none')
            axarr[i,j].yaxis.set_ticks_position('none') 
    plt.savefig('User_ratings.png',dpi=300,format='png')
    con.close()


def get_highest_cos(cos_sim,aid,aid_dict,top_N=5):
    ''' Returns the anime with the highest cosine similarity '''
    closest = sorted(range(len(cos_sim[aid,:])), key=lambda i: cos_sim[aid,i])[-(top_N+1):]
    closest.reverse()
    con = sqlite3.connect(database)
    cur = con.cursor()
    aid_list = []
    sim_list = []
    for idx in range(0,top_N+1):
        aid_2=closest[idx]
        anime = cur.execute('SELECT Name FROM AnimeData where Anime=={0}'.format(aid_dict[aid_2])).fetchone()[0]
        if aid_2==aid:
            print('Recommendations for ' + anime + ':')
        else:
            print(anime + ': ' + str(cos_sim[aid,aid_2]))
            aid_list.append(aid_2)
            sim_list.append(cos_sim[aid,aid_2])
    return aid_list, sim_list


def get_image(anime_id):
    ''' Loads an anime's cover image. Downloads the image if necessary.'''
    con = sqlite3.connect(database)
    cur = con.cursor()
    image_link = cur.execute('SELECT Image FROM AnimeData where Anime=={0}'.format(anime_id)).fetchone()[0]
    image_name = image_link.split('/')[-1]
    try:
        image = Image.open('images/' + image_name)
    except IOError:
        response = requests.get(image_link)
        img = Image.open(StringIO(response.content))
        img.save('images/' + image_name)
        image = Image.open('images/' + image_name)
        print('Saved image for anime_id = ' + str(anime_id))
        time.sleep(0.5)
    con.close()
    return image
    

def plot_top_sims(cos_sim_mat,aids,N_recs):
    ''' Plot of anime with the highest similarity scores. '''
    plt.figure(figsize=(6, 10))
    f, axarr = plt.subplots(len(aids),N_recs+1)
    f.set_size_inches(6, 10)
    f.tight_layout()
    for (idx_0,aid) in enumerate(aids):
        image = get_image(aid_dict[aid])
        axarr[idx_0,0].imshow(image)
        axarr[idx_0,0].axis("off")
        axarr[idx_0,0].set_title('Query ' + str(idx_0+1),size=10)
        top_aids,top_sims = get_highest_cos(cos_sim_mat,aid,aid_dict,N_recs)
        for (idx_1,aid_1) in enumerate(top_aids):
            image = get_image(aid_dict[aid_1])
            axarr[idx_0,idx_1+1].imshow(image)
            axarr[idx_0,idx_1+1].axis("off")
            axarr[idx_0,idx_1+1].set_title('Sim = {:.2f}'.format(top_sims[idx_1]),size=10)
        # Add horizonatal lines:
        if not idx_0==0 or idx_0==len(aids):
            line = lines.Line2D((0,1),(1-1.0/len(aids)*idx_0*0.98,1-1.0/len(aids)*idx_0*0.98), transform=f.transFigure,color=[0,0,0])
            f.lines.append(line)
            # TODO: the 0.98 shouldn't be necessary. Fixit. 


#%% Main code:
# Explore the data a bit:
rating_distributions()
# Create a dictionary for the anime:
aid_dict, aid_dict_inv = get_aid_dict(load=True)
# Get sparse matrix with rows = users, columns = ratings
data = process_user_ratings(aid_dict_inv)




# Choose some specific anime to investigate
death_note_aid =  aid_dict_inv[1535]
code_geass_aid =  aid_dict_inv[1575] 
cowboy_bebop_aid = aid_dict_inv[1]
attack_on_titan_aid = aid_dict_inv[16498]
mirai_nikki_aid = aid_dict_inv[10620]
cicada_aid = aid_dict_inv[934]
anime_ids = [1,1535,1575,16498,10620]
aids = [aid_dict_inv[x] for x in anime_ids]

# Calculate similarity for these guys:
data = data.tocsc()
cos_sim_mat = np.zeros((len(aid_dict),len(aid_dict)),dtype=np.float32)
cutoff = 20
for aid_0 in aids:
    for aid_1 in range(0,len(aid_dict)):
        mult_cols = data[:,aid_0].multiply(data[:,aid_1])
        voting_users = mult_cols.nonzero()[0]
        if len(voting_users) > cutoff:
            numerator = np.sum(mult_cols)
            denominator = np.sqrt(np.sum(data[voting_users,aid_0].power(2))*\
                          np.sum(data[voting_users,aid_1].power(2)))
            cos_sim_mat[aid_0,aid_1] = numerator/denominator

# Plot a histrogram of these similarities:
for aid in aids:
    plt.hist(cos_sim_mat[aid,:])
    plt.show()

# Plot the top similarities:
plot_top_sims(cos_sim_mat,aids,3)







