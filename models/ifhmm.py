"""
Hidden Markov Model
Son N. Tran
"""

import numpy as np
import math

from models.accuracy import pred_accuracy

EPSILON = 0.0000000001

class IFHMM_DIS(object):    
    def __init__(self,dataset):
        if dataset is not None:
            self.dataset = dataset
            self.xsize = xsize = dataset.total_sensor_values()
            self.hsize = hsize = dataset.separate_act_nums()
            self.combined_act_num = dataset.total_combined_acts()
            self.rnum  = rnum = dataset.resident_num()
            
        self.tweights = []
        for i in range(rnum):
            self.tweights.append(np.zeros((self.combined_act_num,hsize[i]),dtype=np.float))
            
        self.eweights = np.zeros((self.combined_act_num,xsize),dtype=np.float)
        #print([self.xsize,self.hsize])
        
    def estimate_params(self):
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
                    self.tweights[r][cmb_prev_act,self.dataset.spr_act_map(curr_act[r],r)] +=1
                    
            # Update emission probability table
            if sensor is not None:
                sensor = self.dataset.sensor_map(sensor)
                self.eweights[cmb_curr_act,sensor] +=1
        #priors: num_of_resident x priors_of_each_resident
        self.prior = self.dataset.get_prior('separate')
        # Laplace Smoothing & Normalize
        self.normalise()
        # Combine the log_prior and log transition
        self.prior,self.tweights = self.combine_log_params()

    def combine_log_params(self):
        combined_acts = self.dataset.get_act_dict()
        _prior = np.zeros((1,self.combined_act_num),dtype=np.float)        
        for j in range(self.combined_act_num):
            for r in range(self.rnum):
                inx = self.dataset.spr_act_map(combined_acts[j][r],r)
                _prior[0,j] += self.prior[r][0,inx]

        _tweights = np.zeros((self.combined_act_num,self.combined_act_num),dtype=np.float)
        for j_ in range(self.combined_act_num):
            for j in range(self.combined_act_num):
                for r in range(self.rnum):
                    curr = combined_acts[j]
                    _tweights[j_,j] += self.tweights[r][j_,self.dataset.spr_act_map(curr[r],r)]
        return _prior,_tweights
    
    def normalise(self):
        for i in range(self.rnum):
            self.prior[i] = (self.prior[i]+EPSILON)/(np.sum(self.prior[i])+EPSILON*self.hsize[i])
            self.prior[i] = np.log2(self.prior[i])
            self.tweights[i] = (self.tweights[i]+EPSILON)/(np.sum(self.tweights[i],axis=1)+EPSILON*self.hsize[i])[:,np.newaxis]
            self.tweights[i] = np.log2(self.tweights[i])
            
        self.eweights = (self.eweights+EPSILON)/(np.sum(self.eweights,axis=1)+EPSILON*self.xsize)[:,np.newaxis]
        self.eweights = np.log2(self.eweights)
    
    def viterbi(self,X):
        # Doing viterbi
        slen = len(X)
        path = np.zeros((slen-1,self.combined_act_num),dtype=np.int)
        #mu_0 = p(o1|a1,a2)p(a1)p(a2)
        #Since only combined acts are considered
        mu  = self.eweights[:,X[0]] + self.prior
        for t in range(1,slen):
            mu = self.tweights + np.reshape(mu,[self.combined_act_num,1])
            #print(mu)
            mx_inds = np.argmax(mu,axis=0)
            mu = np.transpose(np.amax(mu,axis=0))
            mu = mu + self.eweights[:,X[t]]
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
        valid_x,valid_y = self.dataset.valid_dis_sequences()
        pred = self.viterbi(valid_x)
 
        pred = self.dataset.act_rmap(pred)
        valid_acc = pred_accuracy(pred,valid_y)
        return valid_acc

    # SET PARAMS
    def set_prior(self,prior):
        self.prior = np.log2(prior)
        self.hsize = prior.shape[0]
    def set_transitions(self,trans):
        self.tweights = np.log2(trans)
    def set_emmissions(self,emits):
        self.eweights = np.log2(emits)
        self.xsize = emits.shape[1]
    
class IFHMM_VEC(IFHMM_DIS):
    ## Parallel HMM with observation as a single discrete variable
    def __init__(self,dataset):
        if dataset is not None:
            self.dataset = dataset
            self.xsize = xsize = dataset.sensor_num()
            self.hsize = hsize = dataset.separate_act_nums()
            self.combined_act_num = dataset.total_combined_acts()
            self.rnum  = rnum = dataset.resident_num()
            
        self.tweights = []
        for i in range(rnum):
            self.tweights.append(np.zeros((self.combined_act_num,hsize[i]),dtype=np.float))
        
        self.eweights = np.zeros((self.combined_act_num,xsize,2),dtype=np.float)

    def set_vec_type(self):
        pass
        
    def estimate_params(self):
        self.set_vec_type()
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
                    self.tweights[r][cmb_prev_act,self.dataset.spr_act_map(curr_act[r],r)] +=1
            # Update emission probability table
            if sensor is not None:
                sensor = self.dataset.sensor_vec(sensor,self.vec_type)
                self.eweights[cmb_curr_act,range(self.xsize),sensor] += 1
        #priors: act_num x num_residents
        self.prior = self.dataset.get_prior('separate')
        # Normalise & smoothing
        self.normalise()
        # Combine the log_prior and log transition
        self.prior, self.tweights =  self.combine_log_params()
        
    def normalise(self):
        # Laplace Smoothing & Normalize
        for i in range(self.rnum):
            self.prior[i] = (self.prior[i]+EPSILON)/(np.sum(self.prior[i])+EPSILON*self.hsize[i])
            self.prior[i] = np.log2(self.prior[i])
            self.tweights[i] = (self.tweights[i]+EPSILON)/(np.sum(self.tweights[i],axis=1)+EPSILON*self.hsize[i])[:,np.newaxis]
            self.tweights[i] = np.log2(self.tweights[i])
            if np.sum(np.isinf(self.prior[i]))>0 or np.sum(np.isnan(self.prior[i]))>0:
                raise ValueError("prior has inf")
            if np.sum(np.isinf(self.tweights[i]))>0 or np.sum(np.isnan(self.tweights[i]))>0:
                raise ValueError("transition has inf")

            
        self.eweights = (self.eweights+EPSILON)/(np.sum(self.eweights,axis=2)+EPSILON*2)[:,:,np.newaxis]
        self.eweights = np.log2(self.eweights)

        if np.sum(np.isinf(self.eweights))>0 or np.sum(np.isnan(self.eweights))>0:
            raise ValueError("emmision has inf")
            
    def viterbi(self,X):
        # Doing viterbi
        slen = len(X)
        path = np.zeros((slen-1,self.combined_act_num),dtype=np.int)
        #mu_0 = p(o1|a1,a2)p(a1)p(a2)
        #print(self.prior)

        mu  = np.sum(self.eweights[:,range(self.xsize),X[0]],axis=1) + self.prior
        for t in range(1,slen):
            #mu = mu/np.sum(mu)
            #print(self.tweights)
            if np.sum(np.isinf(mu))>0 or np.sum(np.isnan(mu))>0:
                raise ValueError("Numeric Error in Viterbi")
            
            mu = self.tweights + np.reshape(mu,[self.combined_act_num,1])
            #print(mu)
            mx_inds = np.argmax(mu,axis=0)
            mu = np.transpose(np.amax(mu,axis=0))
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
        valid_x,valid_y = self.dataset.valid_vec_sequences(self.vec_type)
        pred = self.viterbi(valid_x)
 
        pred = self.dataset.act_rmap(pred)
        valid_acc = pred_accuracy(pred,valid_y)
        return valid_acc

        
class IFHMM_VEC1(IFHMM_VEC):
    def set_vec_type(self):
        self.vec_type = 1

class IFHMM_VEC2(IFHMM_VEC):
    def set_vec_type(self):
        self.vec_type = 2
        
class IFHMM_VEC3(IFHMM_VEC):
    def set_vec_type(self):
        self.vec_type = 3

        
