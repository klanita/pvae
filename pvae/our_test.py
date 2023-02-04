import torch
from pvae.models.tabular import CSV
import json
from argparse import Namespace
import sys
import pandas as pd
from pvae.pplots import plot_embedding, save_to_file
import matplotlib.pyplot as plt

def load_model(model_name):
    with open(model_name.replace('model.ckpt', 'args.json')) as json_file:
        args = json.load(json_file)
    ns = Namespace(**args)
    model = CSV(ns)
    
    model.load_state_dict(torch.load(model_name))
    model.eval()
   
    return model

def get_predictions(model, features): 
    with torch.no_grad():
        preds = model(torch.Tensor(features))
    
    return preds

def our_test(model_name, family, colours=['short_name', 'tree3'], path='/Users/klanna/UniParis/pvae/data/', losses=None):
    features = pd.read_csv(f'{path}/{family}.csv', header=None).values[:, :-1]
    colors = pd.read_csv(f'{path}/{family}_colours.csv')
    
    model = load_model(model_name)
    preds = get_predictions(model, features)
    embeddings = preds[0].loc.numpy()
    
    for col in colours:
        file_name = model_name.replace('model.ckpt', f'{family}_{col}.png')
        plot_embedding(
                embeddings,
                labels=colors[col],
                labels_text=None,
                labels_idx=None,
                col_dict=None,
                title=col,
                show_lines=False,
                show_text=True,
                show_legend=True,
                axis_equal=True,
                circle_size=30,
                circe_transparency=1.0,
                line_transparency=0.8,
                line_width=0.8,
                fontsize=9,
                fig_width=10,
                fig_height=10,
                file_name=file_name,
                file_format=None,
                labels_name=None,
                width_ratios=[7, 1],
                bbox=(1.3, 0.7),
                plot_legend=True,
                is_hyperbolic=True
            )

    if not (losses is None):
        n = len(losses.keys())
        fig, axes = plt.subplots(n, 1, sharex=True)
        file_name = model_name.replace('model.ckpt', f'losses.png')
        
        for i, l in enumerate(losses):
            axes[i].plot(losses[l])
            axes[i].set_title(l)
            
        save_to_file(fig, file_name)