"""
Mixed Dependencies - Hidden Markov Models for Multi-resident Activity Recognition
Son N. Tran, CSIRO
email: sontn.fz@gmail.com
===========================================================
Models:
    - Mixed HMMs
Sensors' state representation
    - Discrete
    - Vector 1: Vector of sensors' values
    - Vector 2: Binary vector of changing states, i.e 1 for sensor whose state is changed
    - Vector 3: One-hot vector for which sensor is seen when the activities happen
"""
import numpy as np
import math

from models.utils import *

#EPSILON = 0.00001
EPSILON = 0.00000000000000001

class MixHMM(object):
    def __init__(self,dataset,state_type='dis'):
        if dataset is not None:
            self.state_type = state_type
            self.dataset = dataset
            
            self.spr_act_num = spr_act_num = dataset.separate_act_nums()
            self.cmb_act_num = cmb_act_num = dataset.total_combined_acts()
            self.sen_num     = sen_num     = dataset.sensor_num()
            self.sen_val_num = sen_val_num = dataset.total_sensor_values()
            self.rnum        = rnum        = dataset.resident_num()

            self.prior = np.zeros((1,cmb_act_num),dtype=np.float)
            
            self.tweights = np.zeros((cmb_act_num,cmb_act_num),dtype=np.float)
            if state_type=="dis":
                self.xsize = sen_val_num
                self.eweights = np.zeros((cmb_act_num,sen_val_num),dtype=np.float)
            elif state_type=="vec1" or state_type=="vec2" or state_type=="vec3":
                self.xsize = sen_num
                self.eweights = np.zeros((cmb_act_num,sen_num,2),dtype=np.float)
                self.dataset.set_vector_type(int(state_type[-1])) 
            else:
                return
    
    def estimate_params(self):
        _eweights = []
        _spr_tweights = []
        if self.state_type=="dis":
            _eweights = np.zeros((self.cmb_act_num,self.xsize),dtype=float)
        else:
            _eweights = np.zeros((self.cmb_act_num,self.xsize,2),dtype=float)
                
        for r in range(self.rnum):
            _spr_tweights.append(np.zeros((self.cmb_act_num,self.spr_act_num[r]),dtype=float))
            
        _cmb_tweights = np.zeros((self.cmb_act_num,self.cmb_act_num),dtype=float)
        
        # update
        while True:
            prev_act,curr_act,sensor = self.dataset.next()
            # Update transaction probability table
            # tweights[j',j] = p^k(j|j')
            if curr_act is not None:
                cmb_curr_act = self.dataset.act_map(curr_act)
            else:
                break

            if prev_act is not None:
                cmb_prev_act = self.dataset.act_map(prev_act)
                for r in range(self.rnum):
                    # Separate mode
                    _spr_tweights[r][cmb_prev_act,self.dataset.spr_act_map(curr_act[r],r)] +=1
                    
                # Combine mode 
                _cmb_tweights[cmb_prev_act,cmb_curr_act] +=1
               
            # Update emission probability table
            if sensor is not None:
                if self.state_type=="dis":
                    sensor = self.dataset.sensor_map(sensor)
                    _eweights[cmb_curr_act,sensor] +=1
                else:
                    sensor = self.dataset.sensor_vec(sensor)
                    _eweights[cmb_curr_act,range(self.xsize),sensor] +=1

        #priors
        _spr_priors = self.dataset.get_prior(0)
        _cmb_priors = self.dataset.get_prior(1)
        
        # Laplace Smoothing & Normalize
        for r in range(len(_spr_priors)):
            _spr_priors[r] = (_spr_priors[r]+EPSILON)/(np.sum(_spr_priors[r])+EPSILON*self.spr_act_num[r])
            _spr_priors[r] = np.log2(_spr_priors[r])

        _cmb_priors = (_cmb_priors+EPSILON)/(np.sum(_cmb_priors)+EPSILON*self.cmb_act_num)
        _cmb_priors = np.log2(_cmb_priors)
            
        if self.state_type=="dis":
            _eweights = (_eweights+EPSILON)/(np.sum(_eweights,axis=1)+EPSILON*self.xsize)[:,np.newaxis]
                                             
        else:
            _eweights = (_eweights+EPSILON)/(np.sum(_eweights,axis=2)+EPSILON*2)[:,:,np.newaxis]
        _eweights = np.log2(_eweights)

        for r in range(self.rnum):
            _spr_tweights[r] = (_spr_tweights[r]+EPSILON)/(np.sum(_spr_tweights[r],axis=1)+EPSILON*self.spr_act_num[r])[:,np.newaxis]
            _spr_tweights[r] = np.log2(_spr_tweights[r])
        
        _cmb_tweights = (_cmb_tweights+EPSILON)/(np.sum(_cmb_tweights,axis=1)+EPSILON*self.cmb_act_num)
        _cmb_tweights = np.log2(_cmb_tweights)
        
        ####### Combine params ######
        combined_acts = self.dataset.get_act_dict()
        # for prior 
        self.prior = _cmb_priors
        
        for j in range(self.cmb_act_num):
            for r in range(self.rnum):
                inx = self.dataset.spr_act_map(combined_acts[j][r],r)
                self.prior[0,j] += _spr_priors[r][0,inx]
                 
        # for emission
        self.eweights = _eweights

        # for transition
        self.tweights = _cmb_tweights
        
        for j in range(self.cmb_act_num):
            for r in range(self.rnum):
                inx = self.dataset.spr_act_map(combined_acts[j][r],r)
                for j_ in range(self.cmb_act_num):
                    inx_ = self.dataset.spr_act_map(combined_acts[j_][r],r)
                    self.tweights[j_,j] += _spr_tweights[r][j_,inx]
        
                    
    def viterbi(self,X):
        # Doing viterbi
        slen = len(X)
        path = np.zeros((slen-1,self.cmb_act_num),dtype=np.int)
        if self.state_type=="dis":
            mu  = self.eweights[:,X[0]] + self.prior
        else:
            mu  = np.sum(self.eweights[:,range(self.xsize),X[0]],axis=1) + self.prior
            
        for t in range(1,slen):
            mu = self.tweights + np.reshape(mu,[self.cmb_act_num,1])
            mx_inds = np.argmax(mu,axis=0)
            mu = np.transpose(np.amax(mu,axis=0))

            if self.state_type=="dis":
                mu = mu + self.eweights[:,X[t]]
            else:
                mu = mu + np.sum(self.eweights[:,range(self.xsize),X[t]],axis=1)
            path[t-1,:] = mx_inds
        # Extract
        max_inx = np.argmax(mu)
        Y = [-1]*slen
        Y[-1]=max_inx
        for t in range(slen-2,-1,-1):
            max_inx = path[t,max_inx]
            Y[t] = max_inx
        
        return Y
    
    def run(self):
        self.estimate_params()
        valid_acc = 0
        if self.state_type=="dis":
            valid_x,valid_y = self.dataset.valid_dis_sequences()
        else:
            valid_x,valid_y = self.dataset.valid_vec_sequences()
        pred = self.viterbi(valid_x)
 
        pred = self.dataset.act_rmap(pred)

        # Compute accuracy
        valid_acc = pred_accuracy(pred,valid_y)
        return valid_acc, pred, valid_y

    # SET PARAMS:
    def set_prior(self,prior):
        self.prior = np.log2(prior)
        self.hsize = prior.shape[0]
    def set_transitions(self,trans):
        self.tweights = np.log2(trans)
    def set_emmissions(self,emits):
        self.eweights = np.log2(emits)
        self.xsize = emits.shape[1]
        
