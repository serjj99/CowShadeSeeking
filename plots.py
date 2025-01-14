"""
    Author: A. Roger Arnau (ararnnot@posgrado.upv.es)
    Date: April 2024
"""
import torch

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime

from  tqdm import tqdm

def plot_NN_hist(train_loss, val_loss = None, dir = None, save = None, info = ''):
    
    plt.figure()
    
    plt.semilogy(range(1, len(train_loss)+1), train_loss, label = 'Train loss')
    if val_loss is not None:
        plt.semilogy(range(1, len(train_loss)+1), val_loss, label = 'Validation loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss during training{info}')
    plt.legend()
    plt.grid(which='minor')
    
    if save is not None:
        if dir is not None: dir = f'figures/loss/{dir}'
        else: dir = 'figures/loss'
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(f'{dir}/{save}.png')
    #plt.show()
    plt.close()

def plot_result(dataset, input_vars, predicted_shadow,
                save_name = None):
    
    for day in tqdm(dataset['date_day'].unique()):
        
        filter = dataset['date_day'] == day
        date   = dataset[filter]['date']
        time   = dataset[filter]['time'].reset_index(drop=True)
        shadow = dataset[filter]['shadow']
        pred   = predicted_shadow[filter]
    
        plt.figure()
        
        ax1 = plt.gca()
        ax1.plot(date, shadow, label='Real')
        ax1.plot(date, pred, label='Predicted')
        ax1.set_ylabel('Animals in Shadow', fontsize=14)
        ax1.legend(loc='upper left', title='Animals in Shadow',
                   fontsize=12, title_fontsize=12)
        ax1.tick_params(axis='y', labelsize=12)
        
        ax1.set_xlabel('Time', fontsize=14)

        ax2 = ax1.twinx()
        for var in input_vars:
            if var == 'time_float': continue
            if var == 'thi': label = 'Current'
            if var == 'accum_thi': label = 'Day accumulated'
            if var == 'mean_THI_night': label = 'Night mean'
            ax2.plot(date, dataset[filter][var],
                     alpha=0.35, linestyle = '--', label=label)
        
        ax2.set_ylabel('THI', fontsize=14)
        ax2.tick_params(axis='y', labelsize=12)
        ax2.legend(loc='upper right', title='THI',
                   fontsize=12, title_fontsize=12)
        
        num_labels = 7
        x_labels = np.linspace(0, len(time) - 1, num_labels, dtype = int)
        x_labels_str = [str(label)[:-3] for label in time[x_labels]]
        ax1.set_xticks(x_labels)
        ax1.set_xticklabels(x_labels_str, fontsize=12)
        
        if save_name is None:
            plt.show()
        else:
            if not os.path.exists(f'figures/results/{save_name}'):
                os.makedirs(f'figures/results/{save_name}')
            plt.savefig(f'figures/results/{save_name}/{save_name}_pred_{day}.png', bbox_inches='tight', dpi = 300)
        plt.close()
            
            
    # Now the mean            

    plt.figure()
    
    time_intervals = pd.interval_range(
        start = dataset['time_float'].min(),
        end   = dataset['time_float'].max(),
        periods = 30
    )
    
    means = pd.DataFrame(dataset, columns = input_vars + ['shadow'] + ['time'])
    means = means.groupby(pd.cut(means['time_float'], time_intervals))
    mean_shadow = means['shadow'].mean().values
    
    mean_pred = pd.DataFrame()
    mean_pred['time_float'] = dataset['time_float']
    mean_pred['shadow'] = predicted_shadow
    mean_pred = mean_pred.groupby(pd.cut(mean_pred['time_float'], time_intervals)).mean()
    mean_pred = mean_pred['shadow'].values
    
    time_label = means['time'].first().values
    
    plt.plot(time_label, mean_shadow, label='Mean Real Shadow')
    plt.plot(time_label, mean_pred, label='Mean Predicted Shadow')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.twinx()
    for var in input_vars:
        plt.plot(time_label, means[var].mean().values,
                alpha=0.2, label=f'Mean {var}')
    plt.legend()
    
    plt.xlabel('Time')
    plt.ylabel('Mean')
    
    num_labels = 10
    x_labels = np.linspace(0, len(time_label) - 1, num_labels, dtype = int)
    plt.xticks(x_labels, time_label[x_labels], rotation = 45)
    
    if save_name is None:
        plt.show()
    else:
        if not os.path.exists(f'figures/results/{save_name}'):
            os.makedirs(f'figures/results/{save_name}')
        plt.savefig(f'figures/results/{save_name}/{save_name}_mean.png', bbox_inches='tight')
        plt.show()
    plt.close()


def plot_result_v2(dataset, input_vars, predicted_shadow,
                   time_intervals, save_name = None):
    
    results = pd.DataFrame(columns = ['date', 'RMSE'])
    
    for e, day in tqdm(enumerate(dataset['date_day'].unique())):
        
        filter = dataset['date_day'] == day
        date   = dataset[filter]['date']
        time   = dataset[filter]['time'].reset_index(drop=True)
        shadow = dataset[filter]['shadow']
        pred   = predicted_shadow[filter]
    
        plt.figure()
        
        ax1 = plt.gca()
        ax1.plot(date, shadow, label='Real')
        ax1.plot(date, pred, label='Predicted')
        ax1.set_ylabel('Animals in Shadow', fontsize=14)
        ax1.legend(loc='upper left', title='Animals in Shadow',
                   fontsize=12, title_fontsize=12)
        ax1.tick_params(axis='y', labelsize=12)
        
        ax1.set_xlabel('Time', fontsize=14)

        ax2 = ax1.twinx()
        for var in input_vars:
            if var == 'time_float': continue
            if var == 'thi': label = 'Current'
            if var == 'accum_thi': label = 'Day accumulated'
            if var == 'mean_THI_night': label = 'Night mean'
            ax2.plot(date, dataset[filter][var],
                     alpha=0.35, linestyle = '--', label=label)
        
        ax2.set_ylabel('THI', fontsize=14)
        ax2.tick_params(axis='y', labelsize=12)
        ax2.legend(loc='upper right', title='THI',
                   fontsize=12, title_fontsize=12)
        
        num_labels = 7
        x_labels = np.linspace(0, len(time) - 1, num_labels, dtype = int)
        x_labels_str = [str(label)[:-3] for label in time[x_labels]]
        ax1.set_xticks(x_labels)
        ax1.set_xticklabels(x_labels_str, fontsize=12)
        
        if save_name is None:
            plt.show()
        else:
            if not os.path.exists(f'figures/results/{save_name}'):
                os.makedirs(f'figures/results/{save_name}')
            plt.savefig(f'figures/results/{save_name}/{save_name}_pred_{day}.png', bbox_inches='tight', dpi = 500)
        plt.close()
        
        rmse = ((shadow - pred)**2).mean() ** 0.5
        results.loc[e] = [day, rmse]
        
        
    means = pd.DataFrame(dataset, columns = input_vars + ['shadow'] + ['time'])
    means = means.groupby(pd.cut(means['time_float'], time_intervals))
    mean_shadow = means['shadow'].mean().values
    
    mean_pred = pd.DataFrame()
    mean_pred['time_float'] = dataset['time_float']
    mean_pred['pred'] = predicted_shadow
    mean_pred = mean_pred.groupby(pd.cut(mean_pred['time_float'], time_intervals)).mean()
    mean_pred = mean_pred['pred'].values
    
    time_first = [interval.left for interval in time_intervals]    
    means = pd.DataFrame({
        'time': time_first,
        'shadow': mean_shadow,
        'pred': mean_pred
    })
    
    return results, means

def decimal_to_time(decimal_time):
    hours = int(decimal_time)
    minutes = int((decimal_time - hours) * 60)
    return f"{hours:02}:{minutes:02}"

def plot_means(shadow, predicted,  time_intervals,
               save_name = None):
    
    plt.plot(time_intervals, shadow, label='Mean Real Shadow')
    plt.plot(time_intervals, predicted, label='Mean Predicted Shadow')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.xlabel('Time')
    plt.ylabel('Mean')
    
    if False:
        num_labels = 10
        x_labels = np.linspace(0, len(time_intervals) - 1, num_labels, dtype = int)
        txt = [decimal_to_time(t) for t in time_intervals[x_labels]]
        plt.xticks(time_intervals[x_labels], txt, rotation = 45)
    times_x = np.arange(7., 22., 1.5)
    plt.xticks(
        times_x,
        [decimal_to_time(t) for t in times_x],
        rotation = 45
    )
    
    if save_name is None:
        plt.show()
    else:
        if not os.path.exists(f'figures/results/{save_name}'):
            os.makedirs(f'figures/results/{save_name}')
        plt.savefig(f'figures/results/{save_name}/{save_name}_mean.png', bbox_inches='tight', dpi = 500)
        plt.show()
    plt.close()
