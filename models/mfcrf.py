"""
Factorial Conditional Random Field
Son N. Tran
"""

import numpy as np

class MFC(object):
    def __init__(self,conf,dataset,alpha=0,beta=0,gamma=0,state_type='dis'):
        if dataset is not None:
            self.state_type = state_type
            self.dataset = dataset
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            
            self.spr_act_num = spr_act_num = dataset.separate_act_nums()
            self.cmb_act_num = cmb_act_num = dataset.total_combined_acts()
            self.sen_num     = sen_num     = dataset.sensor_num()
            self.sen_val_num = sen_val_num = dataset.total_sensor_values()
            self.rnum        = rnum        = dataset.resident_num()
            
    def estimate_params(self):
            params_count = 2*cmb_act_num           # group feature t=0 & t=T
                        += 2*np.sum(spr_act_num)   # parallel feature t=0 & t=T
                        += cmb_act_num*cmb_act_num # group features
                        += np.sum(np.dot(cmb_act_num,spr_act_num)) # cross features
                        += np.dot(spr_act_num,spr_act_num) # parallel features

            self.params = 0.01*np.random.normal(0,1,(params_count,))

            cost = self.conf.cost
            for iter in range(self.conf.MAX_ITER):
                if iter < self.conf.INITIAL_LR_STEPS:
                    lr = self.conf.lr
                else:
                    lr = self.conf.lr_end

                self.dataset.rewind()
                while not self.dataset.eof:
                    xs,ys = self.dataset.arrrange_batch(self.conf.batch_size)
                    if self.conf.opt == 'SGD':
                        DG = lr*(grads(self.params,xs,ys,K,M,0)+self.conf.cost*self.params)
                    elif self.conf.opt== 'L-BFGS':
                        res = minimize(neg_llh,self.params,jac=grads,method='l-bfgs-b',args=(xs,ys,K,M,cost))
                        self.params = res.x
                        print(res.fun)
                        print(res.message)
                    else:
                        raise ValueError("No optimisation")
                
                    err = self.error()
                    nllh = self.neg_llh()

    def viterbi(self,x):
    
    def error(self):

    def neg_llh(self):
        return llh/dlen    
    
    def run(self):


######################################################################
######################################################################
def grads(params,xs,ys,K,M,cost):
    # Compute gradients of a batch
        
def node_potentials(x,fy0,fyT,fyx):

    return npots

def edge_potentials(fyy):
    # Single CRF: dis-dis
    return np.exp(fyy)

def neg_llh(params,xs,ys,K,M,cost):
    
def forward(npots,epots):
    # forward

    return alpha,logz

def backward(npots,epots):

    return beta

def viterbi(fy0,fyT,fyy,fyx,x):

    return y

def viterbi_(fy0,fyT,fyy,fyx,x):

    return y


def vec2tables(params,K,M):

def tables2vec(fy0,fyT,fyy,fyx):

        
