"""
Parallel Hidden Markov Model
Son N. Tran
"""
from models.hmm import *

import numpy as np

import math


from models.accuracy import pred_accuracy

EPSILON = 0.00001
class SHMM_DIS(HMM_DIS):
    def __init__(self,hsize,xsize):
        self.xsize = xsize
        self.hsize = hsize
        self.tweights = np.zeros((hsize,hsize),dtype=np.float)
        self.eweights = np.zeros((hsize,xsize),dtype=np.float)

    def set_prior(self,prior):
        self.prior = prior;
        
    def add_transition_obs(self,j_,j):
        self.tweights[j_,j] += 1
        
    def add_emission_obs(self,j,i):
        self.eweights[j,i] += 1
        
class PHMM_DIS(object):
    ## Parallel HMM with observation as a single discrete variable
    def __init__(self,dataset):
        self.dataset = dataset
        self.hsize = dataset.total_separate_acts()
        self.xsize = dataset.total_sensor_values()
        self.hmms = [SHMM_DIS(self.hsize,self.xsize), SHMM_DIS(self.hsize,self.xsize)]
        
    def estimate_params(self):
        hmm_num = len(self.hmms);
        while True:
            prev_act,curr_act,sensor = self.dataset.next()
            # Update transaction probability table
            # tweights[j',j] = p^k(j|j')
            if curr_act is None:
                break
            elif prev_act is not None:                
                for k in range(hmm_num):
                    self.hmms[k].add_transition_obs(prev_act[k],curr_act[k])

            # Update emission probability table
            if sensor is not None:
                sensor = self.dataset.sensor_map(sensor)
                for k in range(hmm_num):
                    self.hmms[k].add_emission_obs(curr_act[k],sensor)
        #priors: act_num x num_residents
        priors = self.dataset.get_prior('else')
        for k in range(hmm_num):
            self.hmms[k].set_prior(priors[:,k])

        #Laplace smoothing and normalise
        for k in range(hmm_num):
            self.hmms[k].normalise()
            
    def viterbi(self,X):
        # Viterbi algorithm
        # Input:  X - a sequence of events
        # Output: Y - len x r_num: activities of residents
        hmm_num = len(self.hmms)
        seq_len = len(X)
        Y = [[0]*hmm_num]*seq_len
        for k in range(hmm_num):
            y_ = self.hmms[k].viterbi(X)
            for i in range(seq_len):
                Y[i][k] = y_[i]
        return Y

    def run(self):
        self.estimate_params()
        valid_acc = 0
        valid_x,valid_y = self.dataset.valid_dis_sequences()
        pred = self.viterbi(valid_x)
        valid_acc = pred_accuracy(pred,valid_y)        
        return valid_acc
        

class SHMM_VEC(HMM_VEC):
    def __init__(self,hsize,xsize,vec_type):
        self.xsize = xsize
        self.hsize = hsize
        self.tweights = np.zeros((self.hsize,self.hsize),dtype=np.float)
        self.eweights = np.zeros((self.hsize,self.xsize,2),dtype=np.float)
    def add_transition_obs(self,j_,j):
        self.tweights[j_,j] += 1
        
    def add_emission_obs(self,act,sensor_vec):
        self.eweights[act,range(self.xsize),sensor_vec] += 1
        
        
    def set_prior(self,prior):
        self.prior = prior;
        
class PHMM_VEC(object):
    def __init__(self,dataset):
        self.vec_type = 1
        self.set_vec_type()
        self.dataset = dataset
        self.hsize = dataset.total_separate_acts()
        self.xsize = dataset.sensor_num()

        self.hmms = [SHMM_VEC(self.hsize,self.xsize,self.vec_type),SHMM_VEC(self.hsize,self.xsize,self.vec_type)] 

    def set_vec_type(self):
        pass
    
    def estimate_params(self):
        hmm_num = len(self.hmms);
        while True:
            prev_act,curr_act,sensor = self.dataset.next()
            # Update transaction probability table
            # tweights[j',j] = p^k(j|j')
            if curr_act is None:
                break
            elif prev_act is not None:
                 for k in range(hmm_num):
                    self.hmms[k].add_transition_obs(prev_act[k],curr_act[k])            
            # Update emission probability table
            if sensor is not None:
                sensor = self.dataset.sensor_vec(sensor,self.vec_type)
                for k in range(hmm_num):
                    self.hmms[k].add_emission_obs(curr_act[k],sensor)
    
        #priors: act_num x num_residents
        priors = self.dataset.get_prior('else')
        for k in range(hmm_num):
            self.hmms[k].set_prior(priors[:,k])
        #Laplace smoothing and normalise
        for k in range(hmm_num):
            self.hmms[k].normalise()
          
            
    def viterbi(self,X):
        # Viterbi algorithm
        # Input:  X - a sequence of events
        # Output: Y - len x r_num: activities of residents
        hmm_num = len(self.hmms)
        seq_len = len(X)
        Y = [[0]*hmm_num]*seq_len
        for k in range(hmm_num):
            y_ = self.hmms[k].viterbi(X)
            for i in range(seq_len):
                Y[i][k] = y_[i]
        return Y

    def run(self):
        self.estimate_params()
        valid_acc = 0
        valid_x,valid_y = self.dataset.valid_vec_sequences(self.vec_type)
        pred = self.viterbi(valid_x)
        valid_acc = pred_accuracy(pred,valid_y)        
        return valid_acc
    
class PHMM_VEC1(PHMM_VEC):
    def set_vec_type(self):
        self.vec_type = 1

class PHMM_VEC2(PHMM_VEC):
    def set_vec_type(self):
        self.vec_type = 2
        
class PHMM_VEC3(PHMM_VEC):
    def set_vec_type(self):
        self.vec_type = 3


