# -*- coding: utf-8 -*-
"""
Created on Fri May 05 17:18:56 2017

@author: Chris Pierse
"""
# For tSNE:
import numpy as np
from sklearn.manifold import TSNE
import sqlite3
import pandas as pd
# For bokeh plots:
import bokeh.plotting as bp
from bokeh.models import HoverTool, OpenURL, TapTool
from bokeh.plotting import  show, output_file
from bokeh.embed import components
# Imports from other files:
from processData import get_aid_dict

#%% Key functions:
# Build and fit tsne model:
def tSNE_model(Vt,aid_dict):
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    tsne_V = tsne_model.fit_transform(np.transpose(Vt))
    # Put data in a pandas dataframe:
    tsne_df = pd.DataFrame(tsne_V, columns=['x', 'y'])
    # Save it:
    tsne_df.to_csv('results\\tsne_svd.csv')
    # Get anime names:
    con = sqlite3.connect('user_anime_data.db')
    cur = con.cursor()
    anime_data = cur.execute('SELECT Anime, Name, Score FROM animeData').fetchall()
    anime_data=dict([(x[0],(x[1],x[2])) for x in anime_data])
    anime_names = [anime_data[aid_dict[x]][0] for x in range(Vt.shape[1])]
    anime_scores = [anime_data[aid_dict[x]][1] for x in range(Vt.shape[1])]
    anime_ids = [aid_dict[x] for x in range(Vt.shape[1])]
    tsne_df['anime_name'] = anime_names
    tsne_df['anime_id'] = anime_ids
    tsne_df['rating'] = anime_scores
    return tsne_df

# Plotting the data:
def bokeh_scatter_plot(tsne_df):
    output_file("results\\Anime_similarity.html")
    # Prep plot
    plot_anime_sim = bp.figure(plot_width=700, plot_height=600, title="Anime Similarity plotted with tSNE",
        tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave,tap",
        x_axis_type=None, y_axis_type=None, toolbar_location="below",
        toolbar_sticky=False)
    # Plotting the anime data
    plot_anime_sim.scatter(x='x', y='y', source=tsne_df)
    # Handle hover tools
    hover = plot_anime_sim.select(dict(type=HoverTool))
    hover.tooltips={"Anime":"@anime_name [@rating]"}
    # Add ability to click links:
    url = "http://www.myanimelist.net/anime/@anime_id"
    taptool = plot_anime_sim.select(type=TapTool)
    taptool.callback = OpenURL(url=url)
    # Show the file:
    show(plot_anime_sim)
    
    script, div = components(plot_anime_sim)
    with open("results\\sim_script.js", "w") as text_file:
        text_file.write(script[37:-10])
    with open("results\\sim_html.html", "w") as text_file:
        text_file.write(div)
        
#%% Main code:
if __name__ == "__main__":
    # Load the data:
    aid_dict, aid_dict_inv = get_aid_dict(load=True)
    try:
        Vt = np.load('results\\Vt_50.npy')
    except IOError:
        raise Exception('Vt file not found. Please factorize the ratings matrix.')
    # Calculate tSNE and plot:
    tsne_df = tSNE_model(Vt,aid_dict)
    bokeh_scatter_plot(tsne_df[tsne_df['rating']>0])




