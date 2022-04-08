#!/usr/bin/env python
# coding: utf-8

# In[11]:


import torch.nn as nn
import numpy as np
import pickle
import pprint
import wandb
import torch
import time
import os
import gc
import sys
from tqdm.notebook import tqdm
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .dataset import FaceDataset
from .utils import seed_everything,get_attributes_config
from .metrics import Metrics
from .AverageMeter import AverageMeter
from .EarlyStopping import EarlyStopping

class Trainer:
    def __init__(self,
                 config = None,
                 model = None,
                ):
        self.config    = config
        if self.config.logging:
            self.run = self._start_wandb()
        self.dict_transforms = self.config.transformations
        # Calculating the Model, Scheduler, Optimizer and Loss
        self.model     = model.to(self.config.device)
        self.criterion = self._fetch_loss()
        self.optimizer = self._fetch_optimizer()
        self.scheduler = self._fetch_scheduler()
        # Create output_directory:
        self._start_folders()
    
     
    def train_fn(self,train_loader):
        # Model: train-mode
        if self.config.fp16:
            scaler = GradScaler()
        self.model.train()
        outputs = []
        targets = []
        # Initialize object Average Meter
        losses = AverageMeter()
        tk0 = tqdm(train_loader,total = len(train_loader))
        # Reading batches of data
        for b_idx,data in enumerate(tk0):
            for key,value in data.items():
                data[key] = value.to(self.config.device)
            if self.config.fp16:
                with torch.cuda.amp.autocast():
                    output = self.model(data[self.config.feature_name])
                    loss   = self.criterion(output,data[self.config.target_name].unsqueeze(1))
            else:
                output = self.model(data[self.config.feature_name])
                loss   = self.criterion(output,data[self.config.target_name].unsqueeze(1))
                
            if self.config.accumulation_steps>1:
                loss = loss / self.config.accumulation_steps
            
            if self.config.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
 
            if self.config.max_grad_norm is not None:
                if self.config.fp16:
                    scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            if (b_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.fp16:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Step on every batch not after N accumulations:
            if (self.config.scheduler_params['step_on'] == 'batch')&(self.config.scheduler_params['name'] not in ['Plateu',None]):
                self.scheduler.step()
                    
            # Save Outputs: {Predictions and Target}
            preds  = torch.sigmoid(output)
            outputs.append(preds.cpu().detach().numpy())
            targets.append(data[self.config.target_name].cpu().detach().numpy())
            # Update Loss Bar
            losses.update(loss.detach().item(), train_loader.batch_size)
            tk0.set_postfix(Train_Loss = losses.avg, LR = self.optimizer.param_groups[0]['lr'])
            
        if (self.config.scheduler_params['step_on']=='epoch')&(self.config.scheduler_params['name'] not in ['Plateu',None]):
            self.scheduler.step()
                
        outputs = np.vstack(outputs)
        targets = np.hstack(targets)
        return losses.avg,outputs,targets
    
    def valid_fn(self,valid_loader):
        self.model.eval()
        outputs = []
        targets = []
        # Initialize object Average Meter
        losses = AverageMeter()
        tk0 = tqdm(valid_loader,total = len(valid_loader))
        for b_idx,data in enumerate(tk0):
            for key,value in data.items():
                data[key] = value.to(self.config.device)
            with torch.no_grad():
                output = self.model(data[self.config.feature_name]) # Can be changed if necesary
            preds  = torch.sigmoid(output)
            loss = self.criterion(output,data[self.config.target_name].unsqueeze(1))    # Can be changed if necesary
            targets.append(data[self.config.target_name].cpu().detach().numpy())
            losses.update(loss.detach().item(), valid_loader.batch_size)
            tk0.set_postfix(Eval_Loss=losses.avg)
            # Saving outputs:
            outputs.append(preds.cpu().detach().numpy())
            
        outputs = np.vstack(outputs)
        targets = np.hstack(targets)
        return losses.avg,outputs,targets
    
    def fit(self,path = None):
        # seed everything for reproducibilty
        seed_everything(self.config.seed)
        # Creating datasets for training and Validation
        train_dataset = FaceDataset(path,'train',self.dict_transforms['train'])
        valid_dataset = FaceDataset(path,'test',self.dict_transforms['test'])
        # Creating Dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size  = self.config.batch_size,
                                                   pin_memory  = True,
                                                   num_workers = self.config.num_workers,
                                                   shuffle     = True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size  = self.config.batch_size,
                                                   num_workers = self.config.num_workers,
                                                   shuffle     = False,
                                                   pin_memory  = True)

        self.model.to(self.config.device)
        self.criterion.to(self.config.device)
        es = EarlyStopping (patience = self.config.early_stopping, mode = self.config.mode, delta = 0)
        
        for epoch in range(self.config.epochs):
            print(f'================= EPOCH: {epoch + 1} =================')
            time.sleep(0.3)
            print("**** Training **** ")
            time.sleep(0.3)
            train_loss, train_outputs,train_targets = self.train_fn(train_loader)
            print("**** Validation ****")
            time.sleep(0.3)
            valid_loss, valid_outputs,valid_targets = self.valid_fn(valid_loader)
            # Calculating Metrics
            print("**** Metrics ****")
            time.sleep(0.3)
            valid_metrics   = Metrics(targets = valid_targets,preds = valid_outputs)
            # Saving Metrics:
            if self.config.logging:
                self.run.log({
                    "train_loss" : train_loss,
                    "valid_loss" : valid_loss,
                    "epoch"      : epoch
                })
                self.run.log(valid_metrics)
            if self.config.scheduler_params['step_metric'] != None: 
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(valid_metrics[self.config.scheduler_params['step_metric']]) 
                else:
                    raise Exception('Please choose plateu scheduler')
            # Early Stopping on metrics of evaluation
            es(valid_metrics.get(self.config.es_metric), self.model,self.config.output_path)
            if es.early_stop:
                print('Meet early stopping')
                self._clean_cache()
                if self.config.logging:
                    self.run.log({'best_val_loss':es.get_best_val_score()})
                    self.run.log({'epoch_es':epoch})
                return es.get_best_val_score()
            
        self._clean_cache()
        print("Didn't meet early stopping")
        if self.config.logging:
            self.run.log({'best_val_loss':es.get_best_val_score()})
            self.run.log({'epoch_es':epoch})
        return es.get_best_val_score()
            
        
        
    def _fetch_loss(self):
        '''
        Add any loss that you want:
        '''    
        loss_params = self.config.loss_params
        if loss_params['name'] == 'BCE': return nn.BCEWithLogitsLoss()  
        elif loss_params['name'] == 'CrossEntropy': return nn.CrossEntropyLoss()
        else: raise Exception('Please select a valid loss')
            
        
    def _fetch_scheduler(self):
        '''
        Add any scheduler you want
        '''    
        if self.optimizer is None:
            raise Exception('First choose an optimizer')
        
        else:
            sch_params = self.config.scheduler_params
            
            if sch_params['name'] == 'StepLR':
                return torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                       step_size = sch_parmas['step_size'], 
                                                       gamma     = sch_params.get('gamma',0.1))
            elif sch_params['name'] == 'Plateu': 
                return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                  mode      = self.config.mode, 
                                                                  factor    = sch_params.get('factor',0.1), 
                                                                  patience  = sch_params['patience'], 
                                                                  threshold = 0)
            elif sch_params['name'] == 'CosineAnnealingLR':
                return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                         T_max      = sch_params['T_max'], 
                                         eta_min    = sch_params['min_lr'],
                                         verbose    = True,
                                         last_epoch = -1)
            
            elif sch_params['name'] == 'CosineAnnealingWarmRestarts':
                return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 
                                                                            T_0 = sch_params['T_0'], 
                                                                            T_mult  = 1, 
                                                                            eta_min = sch_params['min_lr'], 
                                                                            last_epoch = -1)
            elif sch_params['name'] == None:
                return None
            else:
                raise Exception('Please choose a valid scheduler')                                 
                
        
    def _fetch_optimizer(self):
        '''
        Add any optimizer you want
        '''
        op_params = self.config.optimizer_params
                                                       
        if op_params['name'] == 'Adam':
            return torch.optim.Adam(self.model.parameters(),lr = self.config.lr, weight_decay = op_params.get('WD',0))
        if op_params['name'] == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr = self.config.lr , weight_decay = op_params.get('WD',0))
        else: 
            raise Exception('Please choose a valid optimizer')
    
    def _clean_cache(self):
        torch.cuda.empty_cache()
        gc.collect()

    def _start_folders(self):
        # Create output folder
        # Calculated:
        self.config.folder_path    = os.path.join(self.config.model_path,self.config.project_name,self.config.runname)
        self.config.output_path    = os.path.join(self.config.folder_path,f'model_{self.config.target_name}.pt')
        os.makedirs(self.config.folder_path,exist_ok = True)
        # Save config file in the output_path
        filename = open(os.path.join(self.config.folder_path,'config.pkl'), "wb")
        pickle.dump(get_attributes_config(self.config), filename)
        filename.close()       
        
    def _start_wandb(self):    
        if self.config.logging:
            run = wandb.init(project = self.config.project_name,
                             config  = get_attributes_config(self.config),
                             save_code = True,
                             reinit    = True)
            run.name = self.config.runname
            run.save()
            return run
        
    def _log(self,input_dict):
        return self.run.log(input_dict)
    
    def getModel(self,best = True):
        if best:
            self.model.load_state_dict(torch.load(self.config.output_path, map_location=self.config.device))
            self.model.eval()
            return self.model
        else:
            return self.model

