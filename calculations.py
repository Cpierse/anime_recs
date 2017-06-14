# -*- coding: utf-8 -*-
"""
Created on Tue May 09 20:07:00 2017

@author: Chris Pierse
"""
import numpy as np
from matplotlib import pyplot as plt
import json, sqlite3
import scipy
import os

# Import useful functions:
import processData
from sgdFactorization import nonzero_factorization

from imp import reload
#%% Data:
# Load relevant data:
aid_dict, aid_dict_inv = processData.get_aid_dict(load=True)
aid_scores = processData.get_avg_aid_scores(aid_dict)
aid_counts = processData.count_aids(aid_dict_inv)

# Choose specific anime to investigate:
anime_ids = [1,1535,1575,16498,10620]
aids = [aid_dict_inv[x] for x in anime_ids]
#%% Key functions:
def predict_ratings(user_rows,cos_sim_mat,threshold=0,top_N=None):
    ''' Predict ratings as weighted sum of sim*ratings '''
    #    if threshold>=0:
    #        cos_sim_mat[cos_sim_mat<threshold]=0
    #    if top_N:
    #        tops = np.arange(cos_sim_mat.shape[1]-(top_N+1),cos_sim_mat.shape[1])
    #        indices = np.argsort(cos_sim_mat,axis=1)
    #        indices = np.delete(indices,tops,axis=1)
    #        for x in zip(indices,range(cos_sim_mat.shape[0])):
    #            cos_sim_mat[x] = 0
    return np.dot(user_rows,cos_sim_mat)/(+1e-10+np.dot(user_rows!=0,cos_sim_mat))

def ratings_error(data,val,cos_sim_mat,threshold=0,batch_size=1024, early_end=True,tol=10**-4,top_N=None):
    ''' Calculate the error in the predict ratings on the validation data set
        using the predict_ratings function '''
    # Get validation data and identify unique users
    val_nz = scipy.sparse.find(val)
    unique_users = np.unique(val_nz[0])
    # Prep cos_sim_mat:
    if threshold>=0:
        cos_sim_mat[cos_sim_mat<threshold]=0
    if top_N:
        tops = np.arange(cos_sim_mat.shape[1]-(top_N+1),cos_sim_mat.shape[1])
        indices = np.argsort(cos_sim_mat,axis=1)
        indices = np.delete(indices,tops,axis=1)
        for x in zip(indices,range(cos_sim_mat.shape[0])):
            cos_sim_mat[x] = 0
    # Go through and calculate the errors in batches
    all_errors = np.empty(0)
    increments = len(unique_users)//batch_size+1
    print('Starting error calculation')
    old_error=10
    for i in range(increments):
        rows = unique_users[i*batch_size:min((i+1)*batch_size,len(unique_users))]
        val_data = val[rows,:].toarray()
        user_data = predict_ratings(data[rows,:].toarray(),cos_sim_mat,threshold,top_N)
        error = np.abs(np.multiply(user_data-val_data,val_data!=0))
        error = error[np.nonzero(error)]
        all_errors= np.append(all_errors,error)
        mean_error,per_comp = np.mean(all_errors),i*1.0/increments
        print('Mean Error: ' + str(mean_error) + ', Percent complete: ' + str(per_comp))
        if early_end:
            if per_comp>0.1 and np.abs(old_error-mean_error)<tol:
                print('Error has converged, ending early')
                break
            old_error=mean_error
    error = [np.mean(all_errors),np.std(all_errors),np.sqrt(np.mean(np.power(all_errors,2)))]
    plt.hist(all_errors)
    plt.title('Ratings errors')
    plt.show()
    return error

def factorization_error(U,Vt,val,batch_size=1024):
    # Get validation data and identify unique users
    val_nz = scipy.sparse.find(val)
    unique_users = np.unique(val_nz[0])
    # Go through and calculate the errors in batches
    all_errors = np.empty(0)
    increments = len(unique_users)//batch_size+1
    print('Starting error calculation')
    for i in range(increments):
        rows = unique_users[i*batch_size:min((i+1)*batch_size,len(unique_users))]
        val_data = val[rows,:].toarray()
        user_data = np.dot(U[rows,:],Vt)
        user_data[user_data>10]=10
        user_data[user_data<0]=0
        error = np.abs(np.multiply(user_data-val_data,val_data!=0))
        error = error[np.nonzero(error)]
        all_errors= np.append(all_errors,error)
        mean_error,per_comp = np.mean(all_errors),i*1.0/increments
        print('Mean Error: ' + str(mean_error) + ', Percent complete: ' + str(per_comp))
    error = [np.mean(all_errors),np.std(all_errors),np.sqrt(np.mean(np.power(all_errors,2)))]
    plt.hist(all_errors)
    plt.title('Ratings errors')
    plt.show()
    return error

def factorize_and_similarity(data,val,load = True,dims=[20,35,50,100,200],subtitle='',
                             eps=0.001, lamb=0,svd_start = True, bias=False,
                             implicit=False):
    error_dict = {}
    for dim in dims:
        fail_flag = False
        error_dict[dim] = {'factorization':[],'ratings':[]}
        print('Working on rank ' + str(dim) + ' matrices')
        # Factorize matrix with stochastic graident descent:
        if load:
            try:
                Vt = np.load(os.path.join('results','Vt_'+str(subtitle)+str(dim)+'.npy'))
                U = np.load(os.path.join('results','U_'+str(subtitle)+str(dim)+'.npy'))
            except IOError:
                fail_flag = True
        if fail_flag or not load:
            U,Vt = nonzero_factorization(data, d=dim, batch_size=64, eps = eps,
                                         eps_drop_frac=0.1, svd_start = svd_start,
                                         lamb=lamb, bias=bias,implicit=implicit)
            np.save(os.path.join('results','Vt_'+str(subtitle)+str(dim)+'.npy'),Vt)
            np.save(os.path.join('results','U_'+str(subtitle)+str(dim)+'.npy'),U)
        # Calculate cosine similarity between item vectors  
        if bias:
            cos_sim_mat = processData.cosine_sim(Vt[:-2,:])
        else:
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
        print('Calculating error in rating predictions on validation set')
        error_rate = ratings_error(data,val,cos_sim_mat)
        print(error_rate)
        error_dict[dim]['ratings'].append(error_rate)
    # Save the results:
    with open(os.path.join('results','factorization_errors_'+subtitle[:-1]+'.json'),'w') as fp:
        json.dump(error_dict,fp)


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

def factorization_results(subtitle='train_',dims = [10,20,25,35,50,75,100,125,150,175,200],
                          val_frac=0.1,test_frac=0.1,lamb=0,implicit=False,
                          svd_start=True):
    print('Loading data')
    try:
        data,val,test = processData.load_data_val_test()
        print(len(data.nonzero()[0]))
    except FileNotFoundError:
        data = processData.process_user_ratings(aid_dict_inv)
        data = remove_zeroscore_anime(data.tolil(),aid_dict_inv)
        print(len(data.nonzero()[0]))
        data,val,test = processData.train_val_test_split(data.tocsr(),val_frac=val_frac,test_frac=test_frac)
        print(len(data.nonzero()[0]))
    
    print('Moving to factorization')
    factorize_and_similarity(data,val,load = True,dims=dims,subtitle=subtitle,
                             lamb=lamb, implicit=implicit,svd_start=svd_start)#nz_')
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
    #factorization_results()
    #factorization_results(True,'train_nz_')
    #factorization_results(subtitle='train_reg-3_',dims = [25,50,100,200],val_frac=0.1,test_frac=0.1,lamb=10**-3)
    #factorization_results(subtitle='train_reg-4_',dims = [25,50,100,200],val_frac=0.1,test_frac=0.1,lamb=10**-4)
    #factorization_results(subtitle='train_reg-2_',dims = [25,50,100,200],val_frac=0.1,test_frac=0.1,lamb=10**-2)
    #factorization_results(subtitle='train_reg-1.5_',dims = [25,50,100,200],val_frac=0.1,test_frac=0.1,lamb=10**-2)

    #factorization_results(subtitle='train_nz_',dims = [25,50,100,200],val_frac=0.1,test_frac=0.1,lamb=0)
    #factorization_results(subtitle='train_reg-2_',dims = [25,50,100,200],val_frac=0.1,test_frac=0.1,lamb=10**-2)

    subtitle = 'train_imp_'
    subtitle = 'train_impreg-3_'
    dims = [50,100, 150, 200]
    val_frac=0.1
    test_frac=0.1
    lamb=10**-3
    implicit=True
    svd_start=True
    load = False
    eps = 10**-4
    bias = False
    dim,d = 50,50
    factorization_results(subtitle=subtitle,dims=dims,val_frac=val_frac,
                          test_frac=test_frac,lamb=lamb,implicit=implicit,
                          svd_start=svd_start)



#%% Extra code for very specific figures:
if __name__ == "test":
    # Plot mean absolute errors vs matrix rank.
    def error_figure():
        dims = [10,20,25,35,50,75,100,125,150,175,200]
        fig, ax = plt.subplots(1,2,figsize=(12, 4))
        subtitles= ['train_','train_nz_']
        titles = ['Prediction error', 'Prediction error (no unreleased anime)']
        for idx in [0,1]:
            error_dict = json.load(open('results\\factorization_errors_'+subtitles[idx][:-1]+'.json','r'))
            ax[idx].plot(dims,np.array([error_dict[str(x)]['factorization'][0] for x in dims])[:,0],'rs',
                                 label='Factorization method')
            ax[idx].plot(dims,np.array([error_dict[str(x)]['ratings'][0] for x in dims])[:,0],'bo',
                                 label='Similarity method')
            ax[idx].set_xticks(range(0,250,25))
            ax[idx].set_yticks(np.arange(0,1.5,0.25))
            ax[idx].set_title(titles[idx])
            ax[idx].legend(loc='lower right', shadow=True,numpoints=1)
            ax[idx].set_xlabel('Matrix rank')
            ax[idx].set_ylabel('Mean absolute error')
        plt.savefig('results\\Error_factorization_train_both'+'.png',dpi=150,format='png')
        plt.show()
    
    ### Sample user prediction: ###
    # Create user:
    user_row = np.zeros((len(aid_dict)))
    user_aids = aids[1:4]
    user_aid_names = ['Death Note', 'Code Geass', 'Shingeki no Kyojin']
    user_row[user_aids] = [10,9,8]
    # Get cosine similarity:
    Vt = np.load('results\\Vt_50.npy')
    cos_sim_mat = processData.cosine_sim(Vt)
    # Estimate ratings:
    pratings = predict_ratings(user_row,cos_sim_mat,threshold=0.2)
    top_aids = np.argsort(pratings)[::-1]
    top_recs = []
    con = sqlite3.connect('user_anime_data.db')
    cur = con.cursor()
    idx = 0
    while len(top_recs)<3 and pratings[top_aids[idx]]>0:
        aid = top_aids[idx]
        rating = pratings[top_aids[idx]]
        idx+=1
        anime_id = aid_dict[aid]
        name,image = cur.execute('SELECT Name, Image FROM AnimeData WHERE Anime={0}'.format(anime_id)).fetchone()
        if np.any([x in name for x in user_aid_names]):
            continue
        top_recs.append((aid,name,anime_id,rating,image))
    user_aid_images = []
    for aid in user_aids:
        name,image = cur.execute('SELECT Name, Image FROM AnimeData WHERE Anime={0}'.format(aid_dict[aid])).fetchone()
        user_aid_images.append(image)
        
    ### Playing with regularization:
    factorization_results(remove_zeros = True,subtitle='train_reg-3_',dims = [50,25,100,200],val_frac=0.1,test_frac=0.1,lamb=10**-3)
    factorization_results(remove_zeros = True,subtitle='train_reg-4_',dims = [25,50,100,200],val_frac=0.1,test_frac=0.1,lamb=10**-4)

