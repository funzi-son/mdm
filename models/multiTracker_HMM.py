'''
Created on 9/2015

@author: fan03d

Run multi-tracker with HMM.
'''
import numpy as np
import preprocess as pp
import simplehmm
import itertools
import hmmtracker

ftraining=['./20150501.csv','./20150502.csv','./20150503.csv','./20150504.csv','./20150505.csv','./20150506.csv']
ftest=['./20150507.csv']

trange =2
dtr=pp.getDataAll(ftraining)
dte=pp.getDataAll(ftest)

results=0.0
rid = 0

print 'Tracker ', rid

hmm1 = hmmtracker.tracker(dtr,rid, trange)

raw_data = pp.getSeqData4Rid(dte, rid, trange)

test_data=[]
test_label=[]
for item in raw_data:
	temp=[]
	templabel=[]
	for item2 in item:
		temp.append(item2[1])
		templabel.append(item2[0])
	test_data.append(temp)
	test_label.append(templabel)

bingo=0
for test_rec,label in itertools.izip(test_data,test_label):
	[state_seq, seq_prob] = hmm1.viterbi(test_rec)
	if state_seq == label:
		bingo +=1

print 'Accuracy ', bingo*1.0/len(test_data)
results += bingo*1.0/len(test_data)

rid = 1

print 'Tracker ', rid

hmm1 = hmmtracker.tracker(dtr,rid, trange)

raw_data = pp.getSeqData4Rid(dte, rid, trange)
test_data=[]
test_label=[]
for item in raw_data:
	temp=[]
	templabel=[]
	for item2 in item:
		temp.append(item2[1])
		templabel.append(item2[0])
	test_data.append(temp)
	test_label.append(templabel)

bingo=0
for test_rec,label in itertools.izip(test_data,test_label):
	[state_seq, seq_prob] = hmm1.viterbi(test_rec)
	if state_seq == label:
		bingo +=1

print 'Accuracy ', bingo*1.0/len(test_data)

print 'Accuracy (all): '
results += bingo*1.0/len(test_data)
results = results/2
print results


print 'Uni-model'

hmm1 = hmmtracker.unitracker(dtr,trange)
results=0.0
bingo=0
total=0
rid=0
raw_data = pp.getSeqData4Rid(dte, rid, trange)

test_data=[]
test_label=[]
for item in raw_data:
	temp=[]
	templabel=[]
	for item2 in item:
		temp.append(item2[1])
		templabel.append(item2[0])
	test_data.append(temp)
	test_label.append(templabel)

for test_rec,label in itertools.izip(test_data,test_label):
	[state_seq, seq_prob] = hmm1.viterbi(test_rec)
	if state_seq == label:
		bingo +=1

total += len(test_data)

rid = 1

raw_data = pp.getSeqData4Rid(dte, rid, trange)
test_data=[]
test_label=[]
for item in raw_data:
	temp=[]
	templabel=[]
	for item2 in item:
		temp.append(item2[1])
		templabel.append(item2[0])
	test_data.append(temp)
	test_label.append(templabel)

for test_rec,label in itertools.izip(test_data,test_label):
	[state_seq, seq_prob] = hmm1.viterbi(test_rec)
	if state_seq == label:
		bingo +=1

total += len(test_data)

results = bingo*1.0/total
print 'Accuracy ',results
