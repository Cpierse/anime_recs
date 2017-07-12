#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:47:45 2017

@author: chris
"""

from flask import Flask, render_template, request, redirect
#from flask_talisman import Talisman

# Imports from my code:
from appCalculations import get_recs,get_names_ids, get_results, get_genre_dict

#%% Definitions:


#%% The flask app:
app = Flask(__name__)
#Talisman(app)

@app.route('/')
def main():
  return redirect('/index')

@app.route('/index',  methods=['GET','POST'])
def index():
    # Load anime names and ids to populate list:
    names_ids = get_names_ids()
    genre_dict = get_genre_dict()
    all_genres = sorted(list(genre_dict.keys()))
    all_genres = list(zip(range(len(all_genres)),all_genres))
    print(request.form)

    if request.method == "POST":
        # Identify filters:
        related, genres = None, None
        if "related" in request.form and request.form['related']=='on':
            related = True
        if "genre" in request.form and request.form['genre']=='on':
            genres = [str(genre_dict[x]).zfill(2) for x in request.form 
                             if x not in 
                             ['genre','related','user_id','anime','rating']]
        # Identify the parameters:
        if "user_id" in request.form:
            user_id = request.form['user_id']
            # Generate the recs and related info:
            recs_plus =  get_recs(user_id=user_id,related=related,genres=genres)
            # Process the recommendations and extra data:
            results = get_results(recs_plus)
        elif "anime" in request.form:
            # Process the request:
            anime = request.form.getlist('anime')
            print(anime)
            names_ids = dict(names_ids)
            anime_ids = [names_ids[x] if x in names_ids else None for x in anime]
            print(anime_ids)
            print(request.form.getlist('rating'))
            user_ratings = dict([x for x in zip(anime_ids,
                                    request.form.getlist('rating'))
                                    if x[0] is not None])
            print(user_ratings)
            # Generate the recs and related info:
            recs_plus =  get_recs(user_ratings=user_ratings,related=related,genres=genres) 
            # Process the recommendations and extra data:
            results = get_results(recs_plus)
        else:
            results=None
    else:
        results = None

    return render_template('index.html',results=results,names_ids=names_ids,
                           genres=all_genres)

if __name__ == '__main__':
    app.run(port=33507)