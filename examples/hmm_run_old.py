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
from models.mixhmm_casas import MixHMM_TEST

from models.utils import save_result
import numpy as np
import os


HOME = os.path.expanduser("~")
EXP_DIR = HOME + "/WORK/experiments/multiresidential/"

data  = {0: CASAS_ADLMR,
         1: ARAS_HouseA,
         2: ARAS_HouseB,
}

path = {0:HOME+"/WORK/Data/CASAS/adlmr/",
        1:HOME+"/WORK/Data/ARAS/",
        2:HOME+"/WORK/Data/ARAS/",
}

fold_num={0:26,
          1:30,
          2:30,
}

models = ["hmm","xhmm","phmm","chmm","fhmm","cd-fhmm","mixhmm","mixhmm_casas"]
states = ["dis","vec1","vec2","vec3"]

DATA_ID  = 0
MODEL_ID = 0
STATE_ID = 0

def main():
    run(7,0,0)
    '''
    for model_id in range(len(models)):
        for state_id in range(len(states)):
            for data_id in range(len(data)):
                if state_id==3 and (data_id==1 or data_id==2):
                    continue
                run(model_id,state_id,data_id)
    '''

def run(model_id,state_id,data_id):
    DATA_PATH = path[data_id]
    result_dir = EXP_DIR + models[model_id] + "/" + states[state_id] + "/" + data[data_id].__name__
    
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
        
    result_log = result_dir+'/log.csv'
    if os.path.isfile(result_log):
        return
        
    FOLD_NUM = fold_num[data_id]
    acc = []
    
    for fold in range(FOLD_NUM):
        dataset = data[data_id](DATA_PATH,fold+1)
        if models[model_id]=='mixhmm':
            hmm = MixHMM(dataset,state_type=states[state_id])
        elif models[model_id]=='mixhmm_casas':
            hmm = MixHMM_TEST(dataset,state_type=states[state_id])
        else:    
            hmm = HMM(dataset,model_type=models[model_id],state_type=states[state_id])

            
        vld_,pred,labs = hmm.run()
        # Save fold result
        save_result(pred,labs,result_dir+'/fold'+str(fold))
        
        acc.append(vld_)

        print(vld_)
        #return
    acc = np.array(acc)
    acc = np.append(acc,[np.mean(acc,axis=0)],axis=0)
    #Save to CSV File
    np.savetxt(result_log,acc,delimiter=',')
    
if __name__=="__main__":
    main()
