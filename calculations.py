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
from sklearn.metrics.pairwise import cosine_similarity


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
anime_ids = [1,1535,1575,16498,10620,840]
aids = [aid_dict_inv[x] for x in anime_ids]
#%% Key functions:
def predict_ratings(user_rows,cos_sim_mat,threshold=0,top_N=None):
    ''' Predict all of a user's ratings as weighted sum of sim*ratings '''
    if threshold>=0:
        cos_sim_mat[cos_sim_mat<threshold]=0
    if not top_N:
        return np.dot(user_rows,cos_sim_mat)/(+1e-10+np.dot(user_rows!=0,cos_sim_mat))
    else:
        results = []
        height,width = user_rows.shape
        for i in range(height):
            valid_anime = user_rows[i]!=0
            specific_cos_sim = np.copy(cos_sim_mat)
            specific_cos_sim[~valid_anime[0,:],:]=0
            indices = np.argsort(np.argsort(specific_cos_sim,axis=0),axis=0)
            specific_cos_sim = np.copy(cos_sim_mat)
            for x in zip(indices,range(cos_sim_mat.shape[0])):
                specific_cos_sim[x[0][:-(top_N+1)],x[1]] = 0
            result = np.dot(np.expand_dims(user_rows[i],axis=0),specific_cos_sim)/(+1e-10+np.sum(specific_cos_sim,axis=0))
            results.append(result)
        return np.array(results)[:,0,:]

def ratings_error_batch(data,val,cos_sim_mat,threshold=0,batch_size=1024, early_end=True,tol=10**-4,top_N=None):
    ''' Calculate the error in the predict ratings on the validation data set
        using the predict_ratings function '''
    # Get validation data and identify unique users
    val_nz = scipy.sparse.find(val)
    unique_users = np.unique(val_nz[0])
    # Prep cos_sim_mat:
    if threshold>=0:
        cos_sim_mat[cos_sim_mat<threshold]=0
    # Go through and calculate the errors in batches
    all_errors = np.empty(0)
    increments = len(unique_users)//batch_size+1
    print('Starting error calculation')
    old_error=10
    for i in range(increments):
        rows = unique_users[i*batch_size:min((i+1)*batch_size,len(unique_users))]
        val_data = val[rows,:].toarray()
        user_data = predict_ratings(data[rows,:].toarray(),cos_sim_mat,-1,top_N)
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

def testing_predict_rating(user_row,anime_idx,cos_sim_col,top_N):
    ''' Predict the rating of a single anime given a user'''
    cos_sim = np.multiply(user_row!=0,cos_sim_col)
    if top_N:
        cos_sim[0,np.argsort(cos_sim)[0,:-(top_N-1)]]=0
    return np.dot(user_row,np.transpose(cos_sim))/(10**-8+np.sum(cos_sim))


def ratings_error_individuals(data,val,cos_sim_mat,threshold=0, early_end=True,tol=10**-4,top_N=None):
    ''' Calculate the error in the predict ratings on the validation data set
        using the predict_ratings function '''
    # Prep cos_sim_mat:
    if threshold>=0:
        cos_sim_mat[cos_sim_mat<threshold]=0
    # Get validation data and identify unique users
    val_nz = scipy.sparse.find(val)
    indices = np.random.permutation(range(val_nz[0].shape[0]))
    val_nz = val_nz[0][indices],val_nz[1][indices],val_nz[2][indices]
    all_errors = []
    idx = 0
    old_error=10
    for user,anime,r_true in zip(val_nz[0],val_nz[1],val_nz[2]):
        r_pred = testing_predict_rating(data[user,:].toarray(),anime,cos_sim_mat[anime,:],top_N)
        all_errors.append(np.abs(r_true-r_pred[0,0]))
        idx +=1
        if idx%10000==9999:
            mean_error,per_comp = np.mean(all_errors),idx*1.0/val_nz[0].shape[0]
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
                             implicit=False, top_N_tests=False):
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
            cos_sim_mat = cosine_similarity(np.transpose(Vt[:-2,:]))
        else:
            cos_sim_mat = cosine_similarity(np.transpose(Vt))
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
        if not top_N_tests:
            print('Calculating error in matrix factorization on validation set')
            error_fact = factorization_error(U,Vt,val)
            print(error_fact)
            error_dict[dim]['factorization'].append(error_fact)
            # Testing
            print('Calculating error in rating predictions on validation set')
            #error_rate = ratings_error_batch(data,val,cos_sim_mat)
            error_rate = ratings_error_individuals(data,val,cos_sim_mat)
            print(error_rate)
            error_dict[dim]['ratings'].append(error_rate)
            with open(os.path.join('results','factorization_errors_'+subtitle[:-1]+'.json'),'w') as fp:
                json.dump(error_dict,fp)
        else:
            error_dict[dim] = {'ratings': {}}
            for top_N in [None,50,20,10,5]:
                error_rate = ratings_error_individuals(data,val,cos_sim_mat,top_N=top_N)
                print(error_rate)
                if top_N:
                    error_dict[dim]['ratings'][top_N]=error_rate
                else:
                    error_dict[dim]['ratings'][None]=error_rate
            with open(os.path.join('results','factorization_errors_'+subtitle+'topN.json'),'w') as fp:
                json.dump(error_dict,fp)
    # Save the results:

def get_valid_anime(aid_dict,aid_dict_inv,min_score=0,min_number=100):
    valid_anime = np.ones(len(aid_dict),dtype='bool')
    con = sqlite3.connect('user_anime_data.db')
    cur = con.cursor()
    anime_data = cur.execute('SELECT Anime FROM animeData WHERE Score<={0} OR \
                             Number<{1} OR Number IS NULL' \
                             .format(min_score,min_number)).fetchall()
    for anime_id in anime_data:
        try:
            aid = aid_dict_inv[anime_id[0]]
            valid_anime[aid] = 0
        except KeyError:
            continue
    con.close()
    return valid_anime


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
    except IOError: #FileNotFoundError:
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

def prep_final_ingredients(dim=50, implicit=True, bias=False, subtitle='imp_',threshold=0):
    ''' Prepare the necessary files for deployment'''
    # Load Data
    data = processData.process_user_ratings(aid_dict_inv)
    data = remove_zeroscore_anime(data.tolil(),aid_dict_inv)
    print(len(data.nonzero()[0]))
    # Factorize the matrix:
    U,Vt = nonzero_factorization(data.tocsr(), d=dim, batch_size=64, eps =10**-4,
                                 eps_drop_frac=0.1, svd_start = True,
                                 lamb=0, bias=bias,implicit=implicit)
    #np.save(os.path.join('results','Vt_'+str(subtitle)+str(dim)+'.npy'),Vt)
    #np.save(os.path.join('results','U_'+str(subtitle)+str(dim)+'.npy'),U)
    # Calculate cosine sim:
    print('Cosine matrix must be calculated on the spot')
    valid_anime = get_valid_anime(aid_dict,aid_dict_inv,min_score=0,min_number=100)
    #np.save(os.path.join('results','valid_anime.npy'),valid_anime)
    # Save to deploy folder:
    np.save(os.path.join('deploy','Vt_'+str(subtitle)+str(dim)+'.npy'),Vt)
    np.save(os.path.join('deploy','valid_anime.npy'),valid_anime)



#%% Main code:
if __name__ == "__main__":
    prep_final_ingredients()





#%% Extra code for very specific figures and tests:
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
    user_aids = aids[1:4]+[aids[5]]
    user_aid_names = ['Death Note', 'Code Geass', 'Shingeki no Kyojin','LotGH']
    user_row[user_aids] = [10,9,8,9]
    # Get cosine similarity:
    #Vt = np.load('results\\Vt_50.npy')
    #cos_sim_mat = processData.cosine_sim(Vt)
    Vt = np.load(os.path.join('results','Vt_imp_50.npy'))
    cos_sim_mat = cosine_similarity(np.transpose(Vt))
    # Estimate ratings:
    #pratings = predict_ratings(user_row,cos_sim_mat,threshold=0.2)
    pratings = np.squeeze([testing_predict_rating(user_row,x,np.expand_dims(cos_sim_mat[x,:],axis=0),20) for x in range(cos_sim_mat.shape[0]) ])
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
    #factorization_results()
    #factorization_results(True,'train_nz_')
    #factorization_results(subtitle='train_reg-3_',dims = [25,50,100,200],val_frac=0.1,test_frac=0.1,lamb=10**-3)
    #factorization_results(subtitle='train_reg-4_',dims = [25,50,100,200],val_frac=0.1,test_frac=0.1,lamb=10**-4)
    #factorization_results(subtitle='train_reg-2_',dims = [25,50,100,200],val_frac=0.1,test_frac=0.1,lamb=10**-2)
    #factorization_results(subtitle='train_reg-1.5_',dims = [25,50,100,200],val_frac=0.1,test_frac=0.1,lamb=10**-2)
    #factorization_results(subtitle='train_nz_',dims = [25,50,100,200],val_frac=0.1,test_frac=0.1,lamb=0)
    #factorization_results(subtitle='train_reg-2_',dims = [25,50,100,200],val_frac=0.1,test_frac=0.1,lamb=10**-2)

    #    subtitle = 'train_imp_'
    #    dims = [50,100, 150, 200]
    #    val_frac=0.1
    #    test_frac=0.1
    #    lamb=0#10**-3
    #    implicit=True
    #    svd_start=True
    #    load = False
    #    eps = 10**-4
    #    bias = False
    #    factorization_results(subtitle=subtitle,dims=dims,val_frac=val_frac,
    #                          test_frac=test_frac,lamb=lamb,implicit=implicit,
    #                          svd_start=svd_start)
