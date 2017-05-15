"""
Hidden Markov Models for Multi-resident Activity Recognition
Son N. Tran, CSIRO
email: sontn.fz@gmail.com
===========================================================
Models:
    - Single HMMs
    - Parallel HMMs
    - Coupled HMMs
    - Factorial HMMs
    - Factorial HMMs with crossed dependencies
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

class HMM(object):
    def __init__(self,dataset,model_type="hmm",state_type='dis'):
        if dataset is not None:
            self.model_type = model_type
            self.state_type = state_type
            self.dataset = dataset
            
            self.spr_act_num = spr_act_num = dataset.separate_act_nums()
            self.cmb_act_num = cmb_act_num = dataset.total_combined_acts()
            self.sen_num     = sen_num     = dataset.sensor_num()
            self.sen_val_num = sen_val_num = dataset.total_sensor_values()
            self.rnum        = rnum        = dataset.resident_num()

            self.prior    = np.zeros((1,cmb_act_num),dtype=np.float)
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
        curr_cmb,prev_cmb,ems_cmb,chsize,phsize,ehsize = self.get_model_config()
        _eweights = []
        _tweights = []
        for r in range(ems_cmb):
            if self.state_type=="dis":
                _eweights.append(np.zeros((ehsize[r],self.xsize),dtype=float))
            else:
                _eweights.append(np.zeros((ehsize[r],self.xsize,2),dtype=float))
                
        for r in range(curr_cmb):
            _tweights.append(np.zeros((phsize[r],chsize[r]),dtype=float))
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
                for r in range(curr_cmb):
                    j_ = cmb_prev_act if prev_cmb==1 else self.dataset.spr_act_map(prev_act[r],r)
                    j  = cmb_curr_act if curr_cmb==1 else self.dataset.spr_act_map(curr_act[r],r) 
                    _tweights[r][j_,j] +=1
               
            # Update emission probability table
            if sensor is not None:
                if self.state_type=="dis":
                    sensor = self.dataset.sensor_map(sensor)
                    for r in range(ems_cmb):
                        j  = cmb_curr_act if ems_cmb==1 else self.dataset.spr_act_map(curr_act[r],r)
                        _eweights[r][j,sensor] +=1
                else:
                    sensor = self.dataset.sensor_vec(sensor)
                    for r in range(ems_cmb):
                        j  = cmb_curr_act if ems_cmb==1 else self.dataset.spr_act_map(curr_act[r],r)
                        _eweights[r][j,range(self.xsize),sensor] +=1

        #priors
        _priors = self.dataset.get_prior(curr_cmb)

        # Laplace Smoothing & Normalize
        for r in range(len(_priors)):
            _priors[r] = (_priors[r]+EPSILON)/(np.sum(_priors[r])+EPSILON*chsize[r])
            _priors[r] = np.log2(_priors[r])

        for r in range(ems_cmb):
            if self.state_type=="dis":
                _eweights[r] = (_eweights[r]+EPSILON)/(np.sum(_eweights[r],axis=1)+EPSILON*self.xsize)[:,np.newaxis]
            else:
                _eweights[r] = (_eweights[r]+EPSILON)/(np.sum(_eweights[r],axis=2)+EPSILON*2)[:,:,np.newaxis]
            _eweights[r] = np.log2(_eweights[r])

        for r in range(curr_cmb):
            _tweights[r] = (_tweights[r]+EPSILON)/(np.sum(_tweights[r],axis=1)+EPSILON*chsize[r])[:,np.newaxis]
            _tweights[r] = np.log2(_tweights[r])

        ####### Combine params ######
        combined_acts = self.dataset.get_act_dict()
        # for prior 
        if curr_cmb==1:
            self.prior = _priors[0]
        else:
            for j in range(self.cmb_act_num):
                for r in range(curr_cmb):
                    inx = self.dataset.spr_act_map(combined_acts[j][r],r)
                    self.prior[0,j] += _priors[r][0,inx]
                    
        # for emission
        if ems_cmb ==1:
            self.eweights = _eweights[0]
        else:
            for j in range(self.cmb_act_num):
                for r in range(ems_cmb):
                    inx = self.dataset.spr_act_map(combined_acts[j][r],r)
                    if self.state_type=="dis":
                        self.eweights[j,:] += _eweights[r][inx,:]
                    else:
                        self.eweights[j,:,:] += _eweights[r][inx,:,:]

        # for transition
        if curr_cmb==1:
            self.tweights = _tweights[0]
        else:
            for j in range(self.cmb_act_num):
                for r in range(curr_cmb):
                    inx = self.dataset.spr_act_map(combined_acts[j][r],r)
                    for j_ in range(self.cmb_act_num):
                        inx_ = self.dataset.spr_act_map(combined_acts[j_][r],r)
                        if prev_cmb ==1: # Cross dependencies
                            self.tweights[j_,j] += _tweights[r][j_,inx]
                        else: # Parallel dependencies
                            self.tweights[j_,j] += _tweights[r][inx_,inx]
                            
    def get_model_config(self):
        curr_cmb = 1
        prev_cmb = 1
        ems_cmb  = 1
        chsize   = [self.cmb_act_num]
        phsize   = [self.cmb_act_num]
        ehsize   = [self.cmb_act_num]
        
        if self.model_type=="hmm":
            curr_cmb = 1
        elif self.model_type=="xhmm":
            ems_cmb  = self.rnum
            ehsize   = self.spr_act_num 
        elif self.model_type=="phmm":
            curr_cmb = self.rnum
            prev_cmb = self.rnum
            ems_cmb  = self.rnum
            chsize   = self.spr_act_num
            phsize   = chsize
            ehsize   = chsize
        elif self.model_type=="chmm":
            curr_cmb = self.rnum
            ems_cmb  = self.rnum
            chsize   = self.spr_act_num
            phsize   = [self.cmb_act_num]*len(chsize)
            ehsize   = self.spr_act_num
        elif self.model_type=="fhmm":
            curr_cmb = self.rnum
            prev_cmb = self.rnum
            chsize   = self.spr_act_num
            phsize   = self.spr_act_num
        elif self.model_type=="cd-fhmm":
            curr_cmb = self.rnum
            chsize   = self.spr_act_num
            phsize   = [self.cmb_act_num]*len(chsize) 
        return curr_cmb,prev_cmb,ems_cmb,chsize,phsize,ehsize

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
        if self.dataset.evaluation_type==1:
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
        elif self.dataset.evaluation_type==2:
            pred_all = []
            y_all = []
            while True:
                if self.state_type=="dis":
                    valid_x,valid_y = self.dataset.next_valid_dis_sequences()
                else:
                    valid_x,valid_y = self.dataset.next_valid_vec_sequences()
                if valid_x is None:
                    break
                pred = self.viterbi(valid_x)
                pred_all.extend(pred)
                y_all.extend(valid_y)
               
            pred_all = self.dataset.act_rmap(pred_all)
            # Compute accuracy
            valid_acc = pred_accuracy(pred_all,y_all)

            ##### For testing
            pred_all = []
            y_all = []
            while True:
                if self.state_type=="dis":
                    test_x,test_y = self.dataset.next_test_dis_sequences()
                else:
                    test_x,test_y = self.dataset.next_test_vec_sequences()
                if test_x is None:
                    break
                pred = self.viterbi(test_x)
                pred_all.extend(pred)
                y_all.extend(test_y)

            pred_all = self.dataset.act_rmap(pred_all)
            # Compute accuracy
            test_acc = pred_accuracy(pred_all,y_all)

            return valid_acc,test_acc, pred_all, y_all
        else:
            raise ValueError('Evaluation type not set')

    # SET PARAMS:
    def set_prior(self,prior):
        self.prior = np.log2(prior)
        self.hsize = prior.shape[0]
    def set_transitions(self,trans):
        self.tweights = np.log2(trans)
    def set_emmissions(self,emits):
        self.eweights = np.log2(emits)
        self.xsize = emits.shape[1]
        
