"""
Test hmms on casas & aras datasets 
Son N. Tran, CSIRO
email: sontn.fz@gmail.com
===========================================================
"""

from data.casas_adlmr_ import *
from data.aras_ import *
from models.hmms import HMM
from models.mixhmm import MixHMM
from models.mdm import MDM

from models.utils import save_result
import numpy as np
import os
import sys


data  = {0: CASAS_ADLMR,
         1: ARAS_HouseA,
         2: ARAS_HouseB,
}

path = {0:"./casas/",
        1:"./aras/",
        2:"./aras/",
}

fold_num={0:26,
          1:30,
          2:30,
}

models = ["hmm","xhmm","phmm","chmm","fhmm","cd-fhmm","md-hmm","mdm"]
states = ["dis","vec1","vec2","vec3"]

DATA_ID  = 0
MODEL_ID = 0
STATE_ID = 0

def main():
    model_name = sys.argv[1]
    data_name  = sys.argv[2]
    state      = sys.argv[3]
    data_args = ["casas","arasa","arasb"]

    if state not in states:
        print("State is not found")
        helpf()
        return
    if model_name not in models:
        print("Model is not found")
        helpf()
        return
    if data_name not in data_args:
        print("Data is not found")
        helpf()
        return
    
    state_id = states.index(state)
    data_id = data_args.index(data_name)
    model_id = models.index(model_name)

    if model_name=="mdm":
        try:
            alpha = int(sys.argv[4])
            beta  = int(sys.argv[5])
            gamma = int(sys.argv[6])
        except IndexError or ValueError:
            print("MDM needs alpha, beta, gamma")
            helpf()
            return
    else: 
        alpha = beta = gamma = 0
        
    if state_id==3 and (data_id==1 or data_id==2):
        raise ValueError("This type of feature cannot be applied to this model");
    acc = run(model_id,state_id,data_id,alpha,beta,gamma)*100

    print("Results of  Model: " + model_name + ", Data:" + data_name + ",State: "+state)
    print("          R1     |     R2     |    All   ")
    for i in range(acc.shape[0]-1):
        print("Day %2d   %.3f   |   %.3f   |   %.3f   " % (i+1,acc[i,0],acc[i,1],acc[i,2]))
    print("==========================================")
    print("Average  %.3f   |   %.3f   |   %.3f   " % (acc[-1,0],acc[-1,1],acc[-1,2]))  
        
def run(model_id,state_id,data_id,alpha,beta,gamma):
    DATA_PATH = path[data_id]
        
    FOLD_NUM = fold_num[data_id]        
    acc = []
    for fold in range(FOLD_NUM):
        dataset = data[data_id](DATA_PATH,fold+1)
        dataset.evaluation_type = 1
        if model_id<6:
            model = HMM(dataset,model_type=models[model_id],state_type=states[state_id])
        if model_id==6:
            model = MDM(dataset,alpha=1,beta=1,gamma=1,state_type=states[state_id])
        if model_id==7:
            model = MDM(dataset,alpha=alpha,beta=beta,gamma=gamma,state_type=states[state_id])
            
        vld_,pred,labs = model.run()
        
        acc.append(vld_)

    acc = np.array(acc)
    acc = np.append(acc,[np.mean(acc,axis=0)],axis=0)

    return acc
    
def helpf():
    print("Usage:")
    print("run.sh <model_name> <data_name> <state_type> [alpha] [beta] [gamma]")
    print(" --- model_name:")
    print(" ---------- hmm     : group-dependency HMMs")
    print(" ---------- xhmm    : grouped-dependency coupled HMMs ")
    print(" ---------- phmm    : parallel HMMs")
    print(" ---------- chmm    : coupled HMMs")
    print(" ---------- fhmm    : factorial HMMs")
    print(" ---------- cd-fhmm : crossed-dependency factorial HMMs")
    print(" ---------- md-hmm  : ensembles of HMMs")
    print(" ---------- mdm     : mixed-dependency model")
    print(" --- data_name:")
    print(" ---------- casas: CASAS")
    print(" ---------- arasa: ARAS House A")
    print(" ---------- arasb: ARAS House B")
    print(" --- state_type: dis , vec1, vec2, vec3")
    print(" --- alpha, beta, gamma: must be set if model_name=mdm")

if __name__=="__main__":
    main()
