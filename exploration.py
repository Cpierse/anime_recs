# -*- coding: utf-8 -*-
"""
Explore the data.
Plot similarities.


Created on Tue May 09 10:09:24 2017

@author: Chris Pierse
"""
import numpy as np
from matplotlib import pyplot as plt

# Import useful functions:
import processData
from sgdFactorization import nonzero_factorization


#%% Data:
# Load relevant data:
aid_dict, aid_dict_inv = processData.get_aid_dict(load=True)
data = processData.process_user_ratings(aid_dict_inv)
aid_counts = processData.count_aids(aid_dict_inv)
aid_scores = processData.get_avg_aid_scores(aid_dict)


# Choose specific anime to investigate:
anime_ids = [1,1535,1575,16498,10620]
aids = [aid_dict_inv[x] for x in anime_ids]

#%% Similarity:
########## Similarity via brute force ##########
data = data.tocsc()
cos_sim_full = np.zeros((len(aid_dict),len(aid_dict)),dtype=np.float32)
cutoff = 20
for aid_0 in aids:
    for aid_1 in range(0,len(aid_dict)):
        mult_cols = data[:,aid_0].multiply(data[:,aid_1])
        voting_users = mult_cols.nonzero()[0]
        if len(voting_users) > cutoff:
            numerator = np.sum(mult_cols)
            denominator = np.sqrt(np.sum(data[voting_users,aid_0].power(2))*\
                          np.sum(data[voting_users,aid_1].power(2)))
            cos_sim_full[aid_0,aid_1] = numerator/denominator
np.save('results\\cos_sim',cos_sim_full)
# Plot the top similarities:
processData.plot_top_sims(cos_sim_full,aids,3,aid_counts,aid_dict)

########## Similarity via matrix factorization ##########
print('Factorizing')
data = data.tocsr()
load=True
for dim in [20,35,50,100,200]:
    print('Working on rank ' + str(dim) + ' matrices')
    # Factorize matrix with stochastic graident descent:
    if not load:
        U,Vt = nonzero_factorization(data, d=dim, batch_size=64, eps = 0.005, svd_start = True)
        np.save('results\\Vt_'+str(dim)+'.npy',Vt)
    else:
        Vt = np.load('results\\Vt_'+str(dim)+'.npy')
    # Calculate cosine similarity between item vectors    
    cos_sim_mat = processData.cosine_sim(Vt)
    # Remove anime that have no score. Anime may have no score if it was not yet released.
    valid_anime = np.dot(np.ones((aid_scores.shape[1],aid_scores.shape[0])),aid_scores)>0
    # Plot top similar animes:
    processData.plot_top_sims(np.multiply(cos_sim_mat,valid_anime),aids,3,
                              aid_counts,aid_dict,img_name = 'results\\Svd_sim_'
                              + str(dim) + '.png',threshold=25)
    # Plot histogram of similarities for the query anime:
    for aid in aids:
        plt.hist(cos_sim_mat[aid,:])
        plt.show()
    # Recommendations could be top sim*score. Calculate and plot this:
    cos_sim_x_scores = np.multiply(np.dot(np.ones((aid_scores.shape[1],aid_scores.shape[0])),aid_scores),cos_sim_mat)
    processData.plot_top_sims(cos_sim_x_scores,aids,3,aid_counts,aid_dict,img_name = 'results\\Svd_sim_'+str(dim)+'x_score.png',threshold=25,var_name='SimXScore')


########## Similarity - determine best dim ##########
data = processData.process_user_ratings(aid_dict_inv)
print(len(data.nonzero()[0]))
data,val,test = processData.train_val_test_split(data,val_frac=0.1,test_frac=0.1)
print(len(data.nonzero()[0]))

data = data.tocsr()
load = False
for dim in [20,35,50,100,200]:
    print('Working on rank ' + str(dim) + ' matrices')
    # Factorize matrix with stochastic graident descent:
    if not load:
        U,Vt = nonzero_factorization(data, d=dim, batch_size=64, eps = 0.005,eps_drop_frac=0.1, svd_start = True)
        np.save('results\\Vt_train'+str(dim)+'.npy',Vt)
    else:
        Vt = np.load('results\\Vt_train'+str(dim)+'.npy')
    # Calculate cosine similarity between item vectors    
    cos_sim_mat = processData.cosine_sim(Vt)
    # Remove anime that have no score. Anime may have no score if it was not yet released.
    valid_anime = np.dot(np.ones((aid_scores.shape[1],aid_scores.shape[0])),aid_scores)>0
    # Plot top similar animes:
    processData.plot_top_sims(np.multiply(cos_sim_mat,valid_anime),aids,3,
                              aid_counts,aid_dict,img_name = 'results\\Svd_sim_train'
                              + str(dim) + '.png',threshold=25)
    # Plot histogram of similarities for the query anime:
    for aid in aids:
        plt.hist(cos_sim_mat[aid,:])
        plt.show()
    # Recommendations could be top sim*score. Calculate and plot this:
    cos_sim_x_scores = np.multiply(np.dot(np.ones((aid_scores.shape[1],aid_scores.shape[0])),aid_scores),cos_sim_mat)
    processData.plot_top_sims(cos_sim_x_scores,aids,3,aid_counts,aid_dict,img_name = 'results\\Svd_sim_train'+str(dim)+'x_score.png',threshold=25,var_name='SimXScore')




