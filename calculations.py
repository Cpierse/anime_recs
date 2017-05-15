# -*- coding: utf-8 -*-
"""
Created on Tue May 09 20:07:00 2017

@author: Chris Pierse
"""
import numpy as np
from matplotlib import pyplot as plt
import json, sqlite3

# Import useful functions:
import processData
from sgdFactorization import nonzero_factorization


#%% Data:
# Load relevant data:
aid_dict, aid_dict_inv = processData.get_aid_dict(load=True)
aid_counts = processData.count_aids(aid_dict_inv)
aid_scores = processData.get_avg_aid_scores(aid_dict)

# Choose specific anime to investigate:
anime_ids = [1,1535,1575,16498,10620]
aids = [aid_dict_inv[x] for x in anime_ids]
#%% Key functions:
def factorization_error(U,Vt,val):
    val_indices = val.nonzero()
    old_user = np.nan
    user_data = []
    errors = []
    for user,aid in zip(val_indices[0],val_indices[1]):
        if user == old_user:
            r_true = val[user,aid]
            r_pred = user_data[aid]
        else:
            old_user = user
            user_data = np.dot(U[user,:],Vt)
            r_true = val[user,aid]
            r_pred = user_data[aid]
        errors.append(np.abs(r_true-r_pred))
    error = [np.mean(errors),np.std(errors),np.sqrt(np.mean(np.power(errors,2)))]
    plt.hist(errors)
    plt.title('Factorization errors')
    plt.show()
    return error
    

def predict_ratings(user_row,cos_sim_mat):
    knowns = np.where(user_row)[0]
    numerator = np.zeros_like(user_row)
    denominator = np.zeros_like(user_row)+1e-8
    for aid in knowns:
        numerator += user_row[aid]*cos_sim_mat[aid,:]
        denominator += cos_sim_mat[aid,:]
    return numerator/denominator

def ratings_error(data,val,cos_sim_mat):
    val_indices = val.nonzero()
    old_user = np.nan
    user_data = []
    errors = []
    for user,aid in zip(val_indices[0],val_indices[1]):
        if user == old_user:
            r_true = val[user,aid]
            r_pred = user_data[aid]
        else:
            old_user = user
            user_data = predict_ratings(data[user,:].toarray()[0],cos_sim_mat)
            r_true = val[user,aid]
            r_pred = user_data[aid]
        errors.append(np.abs(r_true-r_pred))
    error = [np.mean(errors),np.std(errors),np.sqrt(np.power(errors,2))]
    plt.hist(errors)
    plt.title('Ratings errors')
    plt.show()
    return error

def factorize_and_similarity(data,val,load = True,dims=[20,35,50,100,200],subtitle=''):
    error_dict = {}
    for dim in dims:
        fail_flag = False
        error_dict[dim] = {'factorization':[],'ratings':[]}
        print('Working on rank ' + str(dim) + ' matrices')
        # Factorize matrix with stochastic graident descent:
        if load:
            try:
                Vt = np.load('results\\Vt_'+str(subtitle)+str(dim)+'.npy')
                U = np.load('results\\U_'+str(subtitle)+str(dim)+'.npy')
            except IOError:
                fail_flag = True
        if fail_flag or not load:
            U,Vt = nonzero_factorization(data, d=dim, batch_size=64, eps = 0.005,eps_drop_frac=0.1, svd_start = True)
            np.save('results\\Vt_'+str(subtitle)+str(dim)+'.npy',Vt)
            np.save('results\\U_'+str(subtitle)+str(dim)+'.npy',U)
        # Calculate cosine similarity between item vectors    
        cos_sim_mat = processData.cosine_sim(Vt)
        # Remove anime that have no score. Anime may have no score if it was not yet released.
        valid_anime = np.dot(np.ones((aid_scores.shape[1],aid_scores.shape[0])),aid_scores)>0
        # Plot top similar animes:
        processData.plot_top_sims(np.multiply(cos_sim_mat,valid_anime),aids,3,
                                  aid_counts,aid_dict,img_name = 'results\\Svd_sim_'+str(subtitle)
                                  + str(dim) + '.png',threshold=25)
        # Plot histogram of similarities for the query anime:
        for aid in aids:
            plt.hist(cos_sim_mat[aid,:])
            plt.show()
        # Recommendations could be top sim*score. Calculate and plot this:
        cos_sim_x_scores = np.multiply(np.dot(np.ones((aid_scores.shape[1],aid_scores.shape[0])),aid_scores),cos_sim_mat)
        processData.plot_top_sims(cos_sim_x_scores,aids,3,aid_counts,aid_dict,
                                  img_name = 'results\\Svd_sim_'+str(subtitle)+
                                  str(dim)+'x_score.png',threshold=25,
                                  var_name='SimXScore')
        print('Calculating error in matrix factorization on validation set')
        error_fact = factorization_error(U,Vt,val)
        print(error_fact)
        error_dict[dim]['factorization'].append(error_fact)
        # Testing
        print('Calculating error in rating predicitions on validation set')
        cos_sim_mat[cos_sim_mat<0]=0
        error_rate = ratings_error(data,val,cos_sim_mat)
        print(error_rate)
        error_dict[dim]['ratings'].append(error_rate)
    # Save the results:
    json.dump(error_dict,open('results\\factorization_errors_'+subtitle[:-1]+'.json','w'))

def remove_zeroscore_anime(data,aid_dict_inv):
    con = sqlite3.connect('user_anime_data.db')
    cur = con.cursor()
    anime_data = cur.execute('SELECT Anime, Score FROM animeData where Score=0').fetchall()
    for anime_id,score in anime_data:
        try:
            aid = aid_dict_inv[anime_id]
            data[:,aid] = 0
        except KeyError:
            continue
    con.close()
    return data

def factorization_results(remove_zeros=False,subtitle='train_',dims = [10,20,25,35,50,75,100,125,150,175,200]):
    print('Loading data')
    data = processData.process_user_ratings(aid_dict_inv) 
    if remove_zeros:
        data = remove_zeroscore_anime(data.tolil(),aid_dict_inv)
    print(len(data.nonzero()[0]))
    data,val,test = processData.train_val_test_split(data.tocsr(),val_frac=0.1,test_frac=0.1)
    print(len(data.nonzero()[0]))
    
    print('Moving to factorization')
    factorize_and_similarity(data,val,load = True,dims=dims,subtitle='train_')#nz_')
    error_dict = json.load(open('results\\factorization_errors_'+subtitle[:-1]+'.json','r'))
    # Plot the error:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dims,np.array([error_dict[str(x)]['factorization'][0] for x in dims])[:,0],'rs',
                         label='Factorization error')
    ax.plot(dims,np.array([error_dict[str(x)]['ratings'][0] for x in dims])[:,0],'bo',
                         label='Estimated ratings error')
    ax.set_xticks(range(0,250,25))
    ax.set_yticks(np.arange(0,1.5,0.25))
    ax.set_title('Matrix factorization error on validation set')
    ax.legend(loc='lower right', shadow=True,numpoints=1)
    ax.set_xlabel('Matrix rank')
    ax.set_ylabel('Mean absolute error')
    plt.savefig('results\\Error_factorization_'+subtitle[:-1]+'.png',dpi=300,format='png')
    plt.show()

#%% Main code:
if __name__ == "__main__":
    factorization_results()
    factorization_results(True,'train_nz_')
    

#dims = [10,20,25,35,50,75,100,125,150,175,200]
#fig, ax = plt.subplots(1,2,figsize=(12, 4))
#subtitles= ['train_','train_nz_']
#titles = ['Matrix factorization results', 'Matrix factorization results' + ' (cleaned)']
#for idx in [0,1]:
#    error_dict = json.load(open('results\\factorization_errors_'+subtitles[idx][:-1]+'.json','r'))
#    ax[idx].plot(dims,np.array([error_dict[str(x)]['factorization'][0] for x in dims])[:,0],'rs',
#                         label='Factorization error')
#    ax[idx].plot(dims,np.array([error_dict[str(x)]['ratings'][0] for x in dims])[:,0],'bo',
#                         label='Estimated ratings error')
#    ax[idx].set_xticks(range(0,250,25))
#    ax[idx].set_yticks(np.arange(0,1.5,0.25))
#    ax[idx].set_title(titles[idx])
#    ax[idx].legend(loc='lower right', shadow=True,numpoints=1)
#    ax[idx].set_xlabel('Matrix rank')
#    ax[idx].set_ylabel('Mean absolute error')
#plt.savefig('results\\Error_factorization_train_both'+'.png',dpi=300,format='png')
#plt.show()

    
    
