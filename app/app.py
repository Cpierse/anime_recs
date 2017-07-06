#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:47:45 2017

@author: chris
"""

from flask import Flask, render_template, request, redirect

# Imports from my code:
from appCalculations import get_recs,get_names_ids, get_results

#%% Definitions:


#%% The flask app:
app = Flask(__name__)

@app.route('/')
def main():
  return redirect('/index')

@app.route('/index',  methods=['GET','POST'])
def index():
    # Load anime names and ids to populate list:
    names_ids = get_names_ids()
    print(request.form)

    if request.method == "POST":
        # Identify the parameters:
        if "user_id" in request.form:
            user_id = request.form['user_id']
            # Generate the recs and related info:
            recs_plus =  get_recs(user_id=user_id)
            # Process the recommendations and extra data:
            results = get_results(recs_plus)
        elif "anime" in request.form:
            # Process the request:
            anime = request.form.getlist('anime')
            names_ids = dict(names_ids)
            anime_ids = [names_ids[x] if x in names_ids else None for x in anime]
            user_ratings = dict([x for x in zip(anime_ids,
                                    request.form.getlist('rating'))
                                    if x[0] is not None])
            print(user_ratings)
            # Generate the recs and related info:
            recs_plus =  get_recs(user_ratings=user_ratings) 
            # Process the recommendations and extra data:
            results = get_results(recs_plus)
        else:
            results=None
    else:
        results = None

    return render_template('index.html',results=results,names_ids=names_ids)

if __name__ == '__main__':
    app.run(port=33507)