"""
    Author: A. Roger Arnau (ararnnot@posgrado.upv.es)
    Date: April 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import pickle 
import shutil

from tqdm import tqdm
from utils.plots import plot_NN_hist

def fit(model, 
        num_epochs,
        train_dl,
        val_dl      = None,
        loss_fun    = 'MSE',
        optim_name  = 'Adam',
        lr          = 1e-3,
        gamma       = None,
        l2_norm_w   = None,
        display     = True,
        checkpoints = None,
        model_name  = None):
    
    """ Train torch model
    
    Args:
        model (torch model): model to train
        num_epochs (int): number of epochs     
        train_dl (torch data laoder): training data   
        val_dl (torch data laoder): None or validation data (Defoult: None)
        loss_fun (str): torch loss function for trainning (Defoult: 'MSE')
        optim_name (str): tor optimizer (Defoult: 'Adam')
        lr (float): initial learning rate (Defoult: 1e-3)
        gamma (None or float): if not None decreasing lr gamma (Defoult: None)
        l2_norm_w (None or float): if not None, add to loss_fun a l2 weigth (D: None)
        display (boolean): True for displaying progess (Defoult: True)
        checkpoint (None or int): if not None, save model each checkpoint epochs
        model_name (str): used for saving files (None for not save)

    Returns:
        train_loss (list): traning running loss of each epoch
        val_loss (list): validation loss of each epoch ([] otherwise)
    """
    
    # Loss function
    if loss_fun == 'MSE': loss_fn = nn.MSELoss()
    else: raise ValueError(f'loss function {loss_fun} does not exists.')
    
    # Optimizer
    if optim_name == 'Adam':
        if l2_norm_w is not None:
            if l2_norm_w > 0:
                optimizer = optim.Adam(
                    params  = model.parameters(),
                    lr      = lr,
                    weight_decay = l2_norm_w
                ) 
        else:
            optimizer = optim.Adam(
                params  = model.parameters(),
                lr      = lr
            ) 
    else: raise ValueError(f'optimizer {optim} does not exists.')
    
    # (maybe) Decreasing learning rate
    if gamma is not None:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)
        
    # Reestart directory
    dir = f'models/{model_name}'
    if os.path.exists(dir): shutil.rmtree(dir)
    if not os.path.exists(dir): os.makedirs(dir)
    
    train_loss  = []
    val_loss    = []
    t_init      = time.time()

    for epoch in tqdm(range(num_epochs),
                      disable = not display,
                      desc = 'Training epoch'):
        
        # Trainning with train_dl
        
        model.train()
        act_train_loss = 0.0
        total_data     = 0
        
        for batch in train_dl:
            
            x, y = batch
            optimizer.zero_grad()

            outputs = model(x)
            loss    = loss_fn(outputs, y)

            loss.backward()
            optimizer.step()

            # as loss-item() return the mean of the (out - y)^2,
            act_train_loss += loss.item() * x.shape[0]
            total_data     += x.shape[0]
        
        train_loss.append( (act_train_loss / total_data)**0.5 )
        
        
        # Validation with val_dl
        
        if val_dl is not None:
            
            model.eval()
            act_val_loss = 0.0
            total_data   = 0
            
            with torch.no_grad():
                    
                for batch in val_dl:
                    
                    x, y    = batch
                    outputs = model(x)
                    loss    = loss_fn(outputs, y)

                    # as loss-item() return the mean of the (out - y)^2,
                    act_val_loss += loss.item() * x.shape[0]
                    total_data   += x.shape[0]
                
            val_loss.append( (act_val_loss / total_data)**0.5 )
        
        # Trainig current epoch is finished
        
        if gamma is not None:
            scheduler.step()
            
        t_act = time.time() - t_init
        
        if model_name is not None:
            
            label = None
            
            if (epoch + 1) == num_epochs:
                label       = ''
            elif checkpoints is not None:
                if (epoch + 1) % checkpoints == 0:
                    label       = f'. In process: {int(100*(epoch+1)/num_epochs)} %'
                
            if label is not None:
                
                file_name   = f'{model_name}_epoch_{epoch+1}'
                
                plot_NN_hist(train_loss, val_loss,
                             save = f'{model_name}',
                             info = label)
                
                torch.save(model.state_dict(),
                           f'{dir}/{file_name}.pth')
                
                info = {
                    'trainning_date': t_init,
                    'time'          : t_act,
                    'train_loss'    : train_loss,
                    'val_loss'      : val_loss
                }
                with open(f'{dir}/{file_name}_info.pkl', 'wb') as f:
                    pickle.dump(info, f)