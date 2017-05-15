'''
Created on 9/2015

@author: fan03d

A tracker implemented by HMM.
'''
import numpy as np
import preprocess as pp
import simplehmm
import itertools


fsName=['./20150501.csv','./20150502.csv','./20150503.csv','./20150504.csv','./20150504.csv','./20150506.csv', './20150507.csv']

data=pp.getDataAll(fsName)

def tracker(data, rid, trange):

	train_data=[]
	i=0
	while i< len(data):
		pre=[]
		for j in range(trange):
			i=i+j
			if i >= len(data):
				break
			else:
				pre.append((data[i][rid],data[i][2]))

		train_data.append(pre)

	states=['1','2','3','4','5']

	observ=["".join(seq) for seq in itertools.product("01", repeat=5)]


	hmm1 = simplehmm.hmm('Test HMM', states, observ)
	hmm1.train(train_data)
	return hmm1

def unitracker(data, trange):

	train_data=[]
	i=0
	while i< len(data):
		pre=[]
		for j in range(trange):
			i=i+j
			if i >= len(data):
				break
			else:
				pre.append((data[i][0],data[i][2]))
				pre.append((data[i][1],data[i][2]))

		train_data.append(pre)

	states=['1','2','3','4','5']

	observ=["".join(seq) for seq in itertools.product("01", repeat=5)]


	hmm1 = simplehmm.hmm('Test HMM', states, observ)
	hmm1.train(train_data)
	return hmm1
