"""
Hidden Markov Model
Son N. Tran
"""

import numpy as np
import math

from models.accuracy import pred_accuracy

EPSILON = 0.00001

class HMM_DIS(object):
    def __init__(self,dataset):
        if dataset is not None:
            self.dataset = dataset
            self.xsize = xsize = dataset.total_sensor_values()
            self.hsize = hsize = dataset.total_combined_acts()
            self.tweights = np.zeros((hsize,hsize),dtype=np.float)
            self.eweights = np.zeros((hsize,xsize),dtype=np.float)
        #print([self.xsize,self.hsize])
    def estimate_params(self):        
        while True:
            prev_act,curr_act,sensor = self.dataset.next()
            # Update transaction probability table
            # tweights[j',j] = p^k(j|j')
            if curr_act is not None:
                curr_act = self.dataset.act_map(curr_act)
            else:
                break
            
            if prev_act is not None:
                prev_act = self.dataset.act_map(prev_act)
                self.tweights[prev_act,curr_act] +=1
            # Update emission probability table
            if sensor is not None:
                sensor = self.dataset.sensor_map(sensor)
                self.eweights[curr_act,sensor] +=1
        #priors: act_num x num_residents
        self.prior = self.dataset.get_prior()

        # Laplace Smoothing & Normalize
        self.normalise()

    def normalise(self): 
        self.prior = (self.prior+EPSILON)/(np.sum(self.prior)+EPSILON*self.hsize)
        self.tweights = (self.tweights+EPSILON)/(np.sum(self.tweights,axis=1)+EPSILON*self.hsize)[:,np.newaxis]
        self.eweights = (self.eweights+EPSILON)/(np.sum(self.eweights,axis=1)+EPSILON*self.xsize)[:,np.newaxis]

        # Convert to log space
        self.prior = np.log2(self.prior)
        self.tweights = np.log2(self.tweights)
        self.eweights = np.log2(self.eweights)
        
    def viterbi(self,X):
        # Doing viterbi
        slen = len(X)
        path = np.zeros((slen-1,self.hsize),dtype=np.int)
        #mu_0 = p(o1|a1,a2)p(a1)p(a2)
        #print(self.prior)
        mu  = self.eweights[:,X[0]] + self.prior    
        for t in range(1,slen):
            #mu = mu/np.sum(mu)
            #print(self.tweights)
            #print(mu)
            mu = self.tweights + np.reshape(mu,[self.hsize,1])
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
        
    
class HMM_VEC(object):
    ## Parallel HMM with observation as a single discrete variable
    def __init__(self,dataset):
        self.dataset = dataset
        self.hsize = dataset.total_combined_acts()
        self.xsize = dataset.sensor_num()

        self.tweights = np.zeros((self.hsize,self.hsize),dtype=np.float)
        self.eweights = np.zeros((self.hsize,self.xsize,2),dtype=np.float)

    def set_vec_type(self):
        pass
        
    def estimate_params(self):
        self.set_vec_type()
        while True:
            prev_act,curr_act,sensor = self.dataset.next()
            # Update transaction probability table
            # tweights[j',j] = p^k(j|j')
            if curr_act is not None:
                curr_act = self.dataset.act_map(curr_act)
            else:
                break
            
            if prev_act is not None:
                prev_act = self.dataset.act_map(prev_act)
                self.tweights[prev_act,curr_act] +=1
            # Update emission probability table
            if sensor is not None:
                sensor = self.dataset.sensor_vec(sensor,self.vec_type)
                self.eweights[curr_act,range(self.xsize),sensor] += 1
        #priors: act_num x num_residents
        self.prior = self.dataset.get_prior()

        # Normalise & smoothing
        self.normalise()

    def normalise(self):
        # Laplace Smoothing & Normalize
        self.prior = (self.prior+EPSILON)/(np.sum(self.prior)+EPSILON*self.hsize)
        self.tweights = (self.tweights+EPSILON)/(np.sum(self.tweights,axis=1)+EPSILON*self.hsize)[:,np.newaxis]
        self.eweights = (self.eweights+EPSILON)/(np.sum(self.eweights,axis=2)+EPSILON*2)[:,:,np.newaxis]

        # Convert to log space
        self.prior = np.log2(self.prior)
        self.tweights = np.log2(self.tweights)
        self.eweights = np.log2(self.eweights)

        if np.sum(np.isinf(self.prior))>0 or np.sum(np.isnan(self.prior))>0:
            raise ValueError("prior has inf")
        if np.sum(np.isinf(self.tweights))>0 or np.sum(np.isnan(self.tweights))>0:
            raise ValueError("transition has inf")
        if np.sum(np.isinf(self.eweights))>0 or np.sum(np.isnan(self.eweights))>0:
            raise ValueError("emmision has inf")
            
    def viterbi(self,X):
        # Doing viterbi
        slen = len(X)
        path = np.zeros((slen-1,self.hsize),dtype=np.int)
        #mu_0 = p(o1|a1,a2)p(a1)p(a2)
        #print(self.prior)
        
        mu  = np.sum(self.eweights[:,range(self.xsize),X[0]],axis=1) + self.prior
        for t in range(1,slen):
            #mu = mu/np.sum(mu)
            #print(self.tweights)
            if np.sum(np.isinf(mu))>0 or np.sum(np.isnan(mu))>0:
                raise ValueError("Numeric Error in Viterbi")
            
            mu = self.tweights + np.reshape(mu,[self.hsize,1])
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

        
class HMM_VEC1(HMM_VEC):
    def set_vec_type(self):
        self.vec_type = 1

class HMM_VEC2(HMM_VEC):
    def set_vec_type(self):
        self.vec_type = 2
        
class HMM_VEC3(HMM_VEC):
    def set_vec_type(self):
        self.vec_type = 3
