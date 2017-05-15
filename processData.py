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

database = 'user_anime_data.db'
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
            data = scipy.sparse.lil_matrix((max_uid,len(aid_dict_inv)),dtype=scipy.float32)
            for uid in range(0,max_uid):
                user_data = cur.execute('SELECT Anime, Score FROM UserData Where Uid=={0}'.format(uid)).fetchall()
                user_data = np.array([[x[0],x[1]] for x in user_data if x[1]>0])
                if  user_data.shape[0]>0:
                    mean_rating = np.mean(user_data[:,1])
                    data[uid,[aid_dict_inv[x[0]] for x in user_data]]=[scipy.float32(x[1]-mean_rating) for x in user_data]
        else:
            data = scipy.sparse.lil_matrix((max_uid,len(aid_dict_inv)),dtype=scipy.int8)
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
    plt.savefig('results\\User_ratings.png',dpi=300,format='png')
    print([np.mean(all_ratings), np.std(all_ratings)])
    con.close()

def count_aids(aid_dict_inv):
    ''' Counts how many users in the database has rated an anime'''
    con = sqlite3.connect(database)
    cur = con.cursor()  
    counts = cur.execute('SELECT Anime, COUNT(*) FROM UserData WHERE SCORE>0 \
                            GROUP BY Anime').fetchall()
    anime_counts = dict(counts)
    aid_counts = dict([(aid_dict_inv[x[0]],x[1]) for x in anime_counts.items()])
    con.close()
    return aid_counts

def get_avg_aid_scores(aid_dict):
    con = sqlite3.connect(database)
    cur = con.cursor()
    anime_dict = cur.execute('SELECT Anime, Score FROM AnimeData').fetchall()
    anime_dict = dict(anime_dict)
    aid_scores = np.array([[anime_dict[aid_dict[x]] for x in range(len(aid_dict))]])
    con.close()
    return aid_scores


def get_image(anime_id):
    ''' Loads an anime's cover image. Downloads the image if necessary.'''
    con = sqlite3.connect(database)
    cur = con.cursor()
    image_link = cur.execute('SELECT Image FROM AnimeData where Anime=={0}'.format(anime_id)).fetchone()[0]
    image_name = image_link.split('/')[-1]
    if image_name == '0.jpg':
        return None
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

def get_highest_cos(cos_sim,aid,aid_dict,top_N,aid_counts,threshold=0):
    ''' Returns the anime with the highest cosine similarity. Uses rows. '''
    closest = sorted(range(len(cos_sim[aid,:])), key=lambda i: cos_sim[aid,i])
    closest.reverse()
    con = sqlite3.connect(database)
    cur = con.cursor()
    aid_list = []
    sim_list = []
    idx = 0
    while len(aid_list)<top_N:
        aid_2=closest[idx]
        anime = cur.execute('SELECT Name FROM AnimeData where Anime=={0}'.format(aid_dict[aid_2])).fetchone()[0]
        if aid_2==aid:
            print('Recommendations for ' + anime + ':')
        elif aid_2 in aid_counts and aid_counts[aid_2]>threshold:
            print(anime + ': ' + str(cos_sim[aid,aid_2]))
            aid_list.append(aid_2)
            sim_list.append(cos_sim[aid,aid_2])
        idx+=1
    return aid_list, sim_list

def plot_top_sims(cos_sim_mat,aids,N_recs,aid_counts,aid_dict,
                  img_name = 'results\\Sample_similarities.png',threshold=0, 
                  main_name = None, var_name = 'Sim',
                  plot_histograms=True):
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
        top_aids,top_sims = get_highest_cos(cos_sim_mat,aid,aid_dict,N_recs,aid_counts,threshold)
        for (idx_1,aid_1) in enumerate(top_aids):
            image = get_image(aid_dict[aid_1])
            if image != None:
                axarr[idx_0,idx_1+1].imshow(image)
            axarr[idx_0,idx_1+1].axis("off")
            axarr[idx_0,idx_1+1].set_title(var_name + ' = {:.2f}'.format(top_sims[idx_1]),size=10)
        # Add horizonatal lines:
        if not idx_0==0 or idx_0==len(aids):
            line = lines.Line2D((0,1),(1-1.0/len(aids)*idx_0*0.98,1-1.0/len(aids)*idx_0*0.98), transform=f.transFigure,color=[0,0,0])
            f.lines.append(line)
            # TODO: the 0.98 shouldn't be necessary. Fixit. 
    if main_name:
        plt.suptitle(main_name)
    plt.savefig(img_name,dpi=300,format='png')
    plt.show()
    # Plot a histrogram of these similarities:
    if plot_histograms:
        for aid in aids:
            plt.hist(cos_sim_mat[aid,:])
            plt.show()
    return None

def cosine_sim(matrix, columns=True):
    ''' Calculates cosine similarity between columns (for rows, columns = False)'''
    if not columns:
        matrix = np.transpose(matrix)
    square_col_vals = np.array([np.sqrt(np.sum(np.power(matrix,2),axis=0))])
    cos_sim_mat = np.dot(np.transpose(matrix),matrix)/  \
                  np.dot(np.transpose(square_col_vals),square_col_vals)
    return cos_sim_mat

def train_val_test_split(data,val_frac=0.1,test_frac=0.1):
    ''' Splits the data into train, val, test '''
    np.random.seed(777)
    non_zeros = data.nonzero()
    L = len(non_zeros[0])
    val,test = scipy.sparse.lil_matrix(data.shape),scipy.sparse.lil_matrix(data.shape)
    indices = np.random.permutation(range(L))[:np.int(L*(val_frac+test_frac))]
    val_indices = indices[0:np.int(len(indices)*val_frac/(val_frac+test_frac))]
    test_indices = indices[np.int(len(indices)*val_frac/(val_frac+test_frac)):]
    val[non_zeros[0][val_indices],non_zeros[1][val_indices]] = \
        data[non_zeros[0][val_indices],non_zeros[1][val_indices]]
    test[non_zeros[0][test_indices],non_zeros[1][test_indices]] = \
        data[non_zeros[0][test_indices],non_zeros[1][test_indices]]
    data[non_zeros[0][val_indices],non_zeros[1][val_indices]] = 0
    data[non_zeros[0][test_indices],non_zeros[1][test_indices]] = 0
    return data,val,test

#%% Main code:
if __name__ == "__main__":
    # Explore the data a bit:
    rating_distributions()
    # Create a dictionary for the anime:
    aid_dict, aid_dict_inv = get_aid_dict(load=True)
    # Get sparse matrix with rows = users, columns = ratings
    data = process_user_ratings(aid_dict_inv)
    aid_counts = count_aids(aid_dict_inv)
    # Get average score for each anime:
    aid_scores = get_avg_aid_scores(aid_dict)


