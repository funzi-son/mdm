"""
Test mix hmms on casas & aras datasets
Son N. Tran, CSIRO
email: sontn.fz@gmail.com
===========================================================
"""

from data.casas_adlmr_ import *
from data.aras_ import *
from models.hmms import HMM
from models.mixhmm import MixHMM
from models.mdm_da import MDM_DA

from models.utils import save_result
import numpy as np
import os


HOME = os.path.expanduser("~")
EXP_DIR = HOME + "/WORK/experiments/multiresidential/mdm_da/"

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

states = ["dis"]#["dis","vec1","vec2","vec3"]

DATA_ID  = 0
MODEL_ID = 0
STATE_ID = 0

alphas = np.arange(0.8,1.1,0.01)
betas  = np.arange(0.1,0.2,0.01)
gammas = np.arange(-0.1,0.1,0.01)

LEAVE_ONE_OUT = True

def main():
    for data_id in range(0,1):#len(data)):
        if data_id> 0:
            raise ValueError('Only support CASAS')
        
        for state_id in range(0,len(states)):
            if data_id>0 and state_id==3:
                continue
            for alp in alphas:
                for bet in betas:
                    for gam in gammas:
                        run(state_id,data_id,alp,bet,gam)
     
def run(state_id,data_id,alp,bet,gam):
    DATA_PATH = path[data_id]
    if LEAVE_ONE_OUT:
        result_dir = EXP_DIR + "LOO/" + states[state_id] + "/" + data[data_id].__name__ + '/' + str(alp)+'_'+str(bet)+"_"+str(gam)
    else:
        result_dir = EXP_DIR + "model_select/" + states[state_id] + "/" + data[data_id].__name__ + '/' + str(alp)+'_'+str(bet)+"_"+str(gam)
        
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
        
    result_log = result_dir+'/log.csv'
    print(result_log)
    if os.path.isfile(result_log):
        return

    if LEAVE_ONE_OUT:
        FOLD_NUM = fold_num[data_id]
        acc = []
        for fold in range(FOLD_NUM):
            dataset = data[data_id](DATA_PATH,fold+1)
            dataset.evaluation_type = 1
            hmm = MDM_DA(dataset,alpha=alp,beta=bet,gamma=gam,state_type=states[state_id])
            vld_,pred,labs = hmm.run()
            # Save fold result
            #save_result(pred,labs,result_dir+'/fold'+str(fold))
            acc.append(vld_)
            print(vld_)
        #return
        acc = np.array(acc)
        acc = np.append(acc,[np.mean(acc,axis=0)],axis=0)
        #Save to CSV File
        np.savetxt(result_log,acc,delimiter=',')
    else:
        dataset = data[data_id](DATA_PATH)
        dataset.evaluation_type = 2
        hmm = MDM_DA(dataset,alp,bet,gam,state_type=states[state_id])
        vld_,tst_,pred,labs = hmm.run()
        # Save fold result
        #save_result(pred,labs,result_dir+'/')
        print((vld_,tst_))
        rs = []
        rs.append(vld_)
        rs.append(tst_)
        np.savetxt(result_log,rs,delimiter=',')    
    
if __name__=="__main__":
    main()
