"""
Test rnn on casas & aras datasets
Son N. Tran, CSIRO
email: sontn.fz@gmail.com
===========================================================
"""

from data.casas_adlmr_ import *
from data.aras_ import *
from models.rnn import RNN

import numpy as np
import os

EXP_DIR = "/home/tra161/WORK/experiments/multiresidential/"

data  = {0: CASAS_ADLMR,
         1: ARAS_HouseA,
         2: ARAS_HouseB,
}

path = {0:"/home/tra161/WORK/Data/CASAS/adlmr/",
        1:"/home/tra161/WORK/Data/ARAS/",
        2:"/home/tra161/WORK/Data/ARAS/",
}

fold_num={0:26,
          1:30,
          2:30,
}

models = ["rnn"]
states = ["vec1","vec2","vec3"]

DATA_ID  = 0
MODEL_ID = 0
STATE_ID = 0

class Config():
    lr = 0.3
    hidNum = 500
    MAX_ITER = 1000
    NUM_DEC_4_LR_DECAY = 40 # Number of performance reduce before learning rate decay
    MAX_LR_DECAY = 10       # Number of learning rate decay -> for early/late stopping
    LR_DECAY_VAL = 0.5
    batch_size = 1

    weight_decay = 0.01
    
    opt  =  'GD'
    cost = 0.003
    sparse_input = False
def main():
    run(0,0,0,0.3)
    '''
    for model_id in range(len(models)):
        for state_id in range(len(states)):
            for data_id in range(len(data)):
                if state_id==3 and (data_id==1 or data_id==2):
                    continue
                run(model_id,state_id,data_id)
     '''         
def run(model_id,state_id,data_id,lr):    
    DATA_PATH = path[data_id]
    result_dir = EXP_DIR + models[model_id] + "/" + states[state_id] + "/" + data[data_id].__name__
    
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
        
    result_log = result_dir+'/log.csv'
    if os.path.isfile(result_log):
        return
        
    FOLD_NUM = fold_num[data_id]
    acc = []

    conf = Config()
    conf.lr = lr
    for fold in range(FOLD_NUM):
        conf.ckp_file= result_dir + '/fold_'+str(fold+1)+'.ckpt'
        dataset = data[data_id](DATA_PATH,fold+1)
        dataset.set_vector_type(int(states[state_id][-1]))
        rnn = RNN(conf,dataset,model_type=models[model_id])
        vld_,pred,labs = rnn.run()
        # Save fold result
        save_result(pred,labs,result_log+'/fold'+str(fold))
        
        acc.append(vld_)

        print(vld_)
        #return
    acc = np.array(acc)
    acc = np.append(acc,[np.mean(acc,axis=0)],axis=0)
    #Save to CSV File
    np.savetxt(result_log,acc,delimiter=',')
    
if __name__=="__main__":
    main()
