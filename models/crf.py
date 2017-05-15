"""
Conditional Random Field
Son N. Tran
"""
import numpy as np
from scipy.optimize import *
import math

from models.utils import pred_accuracy

class CRF(object):
    def __init__(self,conf,dataset):
        self.conf = conf
        self.dataset = dataset
        self.params = []
                
    def estimate_params(self):
        self.K = K =  self.dataset.total_combined_acts()
        self.M = M = self.dataset.total_sensor_values()
        self.params = 0.01*np.random.normal(0,1,(2*K + K*K + K*M,))

        cost = self.conf.cost
        for iter in range(self.conf.MAX_ITER):
            if iter < self.conf.INITIAL_LR_STEPS:
                lr  = self.conf.lr
            else:
                lr = self.conf.lr_end
                
            self.dataset.rewind()
            while not self.dataset.eof:
                xs,ys   = self.dataset.arrange_batch(self.conf.batch_size)
                #print(len(xs))
                #input('')
                if self.conf.opt=='SGD':
                    DG =  lr*(grads(self.params,xs,ys,K,M,0)+self.conf.cost*self.params)
                    self.params -= DG 
                    #print(DG)
                elif self.conf.opt=='L-BFGS':
                    res = minimize(neg_llh,self.params,jac=grads,method='l-bfgs-b',args=(xs,ys,K,M,cost))
                #   res = minimize(neg_llh, self.params, jac=grads, method='BFGS', args=(xs, ys,K,M))
                    self.params = res.x
                    print(res.fun)
                    print(res.message)
                else:    
                    raise ValueError("No optimisation")
            # TODO Debug end-of-file
            #err = self.error()
            #nllh = self.neg_llh()
            #print('Iter %d nllh = %.5f acc1 = %.5f acc2 = %.5f acc_all = %.5f'%(iter,nllh,err[0],err[1],err[2]))
    def viterbi(self,x):
        fy0,fyT,fyy,fyx = vec2tables(self.params,self.K,self.M)
        return viterbi_(fy0,fyT,fyy,fyx,x)
    
    def error(self):
        fy0,fyT,fyy,fyx=  vec2tables(self.params,self.K,self.M)
        acc = []
        self.dataset.rewind()
        while not self.dataset.eof:
            x,y = self.dataset.arrange_batch(1)
            x = x[0]
            y = y[0]
            pred = viterbi_(fy0,fyT,fyy,fyx,x)
            pred = self.dataset.act_rmap(pred)
            acc_ = pred_accuracy(pred,self.dataset.act_rmap(y))
            acc.append(acc_)
            #print(acc_)
        return np.mean(acc,axis=0)

    def neg_llh(self):
        fy0,fyT,fyy,fyx = vec2tables(self.params,self.K,self.M)
        llh = 0
        dlen = 0
        self.dataset.rewind()
        while not self.dataset.eof:
            x,y = self.dataset.arrange_batch(1)
            x = x[0]
            y = y[0]
            dlen +=1
            slen = len(x)
            
            npots = node_potentials(x,fy0,fyT,fyx)
            epots = edge_potentials(fyy)
            _,logZ = forward(npots,epots)
        
            llh_ = fy0[0,y[0]] + fyx[y[0],x[0]]
            for t in range(1,slen):
                llh_ += (fyy[y[t-1],y[t]] + fyx[y[t],x[t]])

            llh_ += fyT[0,y[-1]]
            llh += (-llh_  + logZ)
            
        return llh/dlen    
    
    def run(self):
        self.estimate_params()
        valid_x,valid_y = self.dataset.valid_dis_sequences()
        pred = self.viterbi(valid_x)
        pred = self.dataset.act_rmap(pred)
        valid_acc = pred_accuracy(pred,valid_y)
        return valid_acc#,pred,valid_y

def grads(params,xs,ys,K,M,cost):
    # Compute gradients of a batch
    fy0,fyT,fyy,fyx = vec2tables(params,K,M)
    
    fy0_grad = np.zeros(fy0.shape,dtype=np.float64)
    fyT_grad = np.zeros(fyT.shape,dtype=np.float64)
    fyy_grad = np.zeros(fyy.shape,dtype=np.float64)
    fyx_grad = np.zeros(fyx.shape,dtype=np.float64)
    
    dlen = len(xs)

    for i in range(dlen):
        x = xs[i]
        y = ys[i]

        slen = len(x)
        npots = node_potentials(x,fy0,fyT,fyx)
        epots = edge_potentials(fyy)

        # forward, backward
        alpha,Z = forward(npots,epots)
        beta    = backward(npots,epots)
        
        t = 0
        nbelief  = np.multiply(alpha[t,:],beta[:,t])
        nbelief  = nbelief/np.sum(nbelief)
        
        fy0_grad[0,y[t]] -= 1
        fy0_grad += nbelief

        fyx_grad[y[t],x[t]] -= 1
        fyx_grad[:,x[t]] += nbelief
        for t in range(1,slen):
            # compute gradient
            nbelief = np.multiply(alpha[t,:],beta[:,t])
            nbelief = nbelief/np.sum(nbelief)
            fyx_grad[y[t],x[t]] -= 1
            fyx_grad[:,x[t]]    += nbelief

            ebelief = (epots*npots[:,t])*np.reshape(alpha[t-1,:],[K,1]).dot(np.reshape(beta[:,t],[1,K]))
            ebelief = ebelief/np.sum(ebelief)
            ##### Debug
            diff_ = np.sum(ebelief,axis=0) - nbelief
            if np.mean(np.abs(diff_))/np.mean(np.abs(nbelief)) > 0.0000001:
                raise ValueError('Value error')
            #####
            
            fyy_grad[y[t-1],y[t]] -= 1
            fyy_grad += ebelief
       
        fyT_grad[0,y[-1]] -= 1
        fyT_grad += nbelief
       
        fy0_grad = fy0_grad + cost*fy0
        fyT_grad = fyT_grad + cost*fyT
        fyy_grad = fyy_grad + cost*fyy
        fyx_grad = fyx_grad + cost*fyx
        
    return tables2vec(fy0_grad,fyT_grad,fyy_grad,fyx_grad)
        
def node_potentials(x,fy0,fyT,fyx):
    # Single CRF: dis-dis
    dlen = len(x)
    K = fyx.shape[0]
    npots = np.zeros((K,dlen),dtype=np.float64)
    for t in range(dlen):
        npots[:,t] = np.exp(fyx[:,x[t]])
    npots[:,0]  = npots[:,0]*np.exp(fy0)
    npots[:,-1] = npots[:,-1]*np.exp(fyT)
    return npots

def edge_potentials(fyy):
    # Single CRF: dis-dis
    return np.exp(fyy)

def neg_llh(params,xs,ys,K,M,cost):
    fy0,fyT,fyy,fyx = vec2tables(params,K,M)
    llh = 0
    dlen = len(xs)
    for i in range(dlen):
        x = xs[i]
        y = ys[i]
    
        slen = len(x)
        npots = node_potentials(x,fy0,fyT,fyx)
        epots = edge_potentials(fyy)
        
        _,logZ = forward(npots,epots)
        
        llh_ = fy0[0,y[0]] + fyx[y[0],x[0]]
        for t in range(1,slen):
            llh_ += (fyy[y[t-1],y[t]] + fyx[y[t],x[t]])

        llh_ += fyT[0,y[-1]]
        llh += (-llh_  + logZ)

    return llh + (cost/2)*(np.sum(np.power(fy0,2)) + np.sum(np.power(fyT,2)) + np.sum(np.power(fyy,2)) + np.sum(np.power(fyx,2)))
    
def forward(npots,epots):
    # forward
    slen = npots.shape[1]
    alpha = np.zeros((slen,npots.shape[0]),dtype=np.float64)
    z     = np.zeros((slen,),dtype=np.float64)
    
    alpha[0,:] = npots[:,0]
    z[0] = np.sum(alpha[0,:])
    alpha[0,:] = alpha[0,:]/z[0]
    for t in range(1,slen):
        alpha[t,:] = alpha[t-1,:].dot(epots*npots[:,t])
        z[t] = np.sum(alpha[t,:])
        alpha[t,:] = alpha[t,:]/z[t]
    # partition
    logz = np.sum(np.log(z))

    return alpha,logz

def backward(npots,epots):
    slen = npots.shape[1]
    beta = np.zeros((epots.shape[0],slen),dtype=np.float64)
    beta[:,slen-1] = np.ones((beta.shape[0],),dtype=np.float64)
    for t in range(slen-2,-1,-1):
        beta[:,t] = (epots*npots[:,t+1]).dot(beta[:,t+1])
        beta[:,t] = beta[:,t]/np.sum(beta[:,t])
    return beta

def viterbi(fy0,fyT,fyy,fyx,x):
    slen = len(x)
    K = fyy.shape[0]
    trace = np.zeros((slen,K),dtype=np.int)

    mu = np.transpose(fy0 + fyx[:,x[0]])
    for t in range(1,slen-1):
        mu = (fyy + fyx[:,x[t]]) + mu
        trace[t-1,:] = np.argmax(mu,axis=0)
        mu = np.amax(mu,axis=0)[:,np.newaxis]
        
        
    mu = (fyT + fyx[:,x[-1]] + fyy) + mu
    trace[slen-2,:] = np.argmax(mu,axis=0)
    mu = np.amax(mu,axis=0)
    
    # Extract
    y = [0]*slen
    max_inx = np.argmax(mu)
    y[-1] = max_inx 
   
    for t in range(slen-2,-1,-1):
        max_inx = trace[t,max_inx] 
        y[t] = max_inx

    return y

def viterbi_(fy0,fyT,fyy,fyx,x):
    slen = len(x)
    K = fyy.shape[0]
    trace = np.zeros((slen,K),dtype=np.int)
    npots = node_potentials(x,fy0,fyT,fyx)
    epots = edge_potentials(fyy)
    

    alpha = np.zeros((slen,K),dtype=np.float64)
    z     = np.zeros((slen,),dtype=np.float64)
    alpha[0,:] = npots[:,0]
    z[0] = np.sum(alpha[0,:])
    alpha[0,:] = alpha[0,:]/z[0]
    for t in range(1,slen):
        mu = (alpha[t-1,:][:,np.newaxis])*epots
        trace[t,:] = np.argmax(mu,axis=0)
        mu = np.amax(mu,axis=0)
        alpha[t,:] = npots[:,t]*mu
        z[t] = np.sum(alpha[t,:])
        alpha[t,:] = alpha[t,:]/z[t]

    y = [-1]*slen
    max_inx= np.argmax(alpha[-1,:])
    y[-1] = max_inx
    for t in range(slen-2,-1,-1):
        max_inx = trace[t+1,max_inx]
        y[t] = max_inx
    return y


def vec2tables(params,K,M):
    fy0 = params[0:K].reshape((1,K))
    fyT = params[K:2*K].reshape((1,K))
    fyy = np.reshape(params[2*K:(2*K+K*K)],[K,K])
    fyx = np.reshape(params[2*K+K*K:2*K+K*K+K*M],[K,M])
    return fy0,fyT,fyy,fyx

def tables2vec(fy0,fyT,fyy,fyx):
    params = np.append(fy0.flatten(),fyT.flatten())
    params = np.append(params,fyy.flatten())
    params = np.append(params,fyx.flatten())
    return params
