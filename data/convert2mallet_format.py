'''
Convert data to mallet format
'''

import os
import numpy as np
from data.casas_adlmr_ import *
from data.aras_ import *

HOME = os.path.expanduser("~")
D_DIR = {0:HOME + "/WORK/Data/CASAS/adlmr/",
           1:HOME + "/WORK/Data/ARAS/House_A/",
           2:HOME + "/WORK/Data/ARAS/House_B/"
           }

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

data_id = 0
def main():
    DATA_PATH = path[data_id]
    d_dir = D_DIR[data_id] + '/mallet/'
    if not os.path.isdir(d_dir):
        os.makedirs(d_dir)

    dataset = data[data_id](DATA_PATH)
    sensor_size = dataset.sensor_num()
    rnum = dataset.resident_num()
    
    ftrain_writer = open(d_dir+"1_16.txt","a")
    next_file = False
    while True:
        prev_act,curr_act,sensor = dataset.next()
        if prev_act is None and curr_act is None and sensor is None:
            break
        
        if prev_act is None and next_file:
            ftrain_writer.write('\n')
        else:
            next_file = True
            
        if data_id ==0: #CASAS
            dataset.set_vector_type(3)
        elif data_id ==1: # ARAS House A
            dataset.set_vector_type(1) 
        else:   # ARAS House B
            dataset.set_vector_type(1)
    
        sensor = dataset.sensor_vec(sensor)
        dstr = ""
        for r in range(rnum):
            dstr+=str(dataset.spr_act_map(curr_act[r],r))+" "
        dstr+="----"
        for f in sensor:
            dstr += " " + str(f)
            
        ftrain_writer.write(dstr+'\n')
        
    ftrain_writer.close()
    
    fvalid_writer = open(d_dir+"17_21.txt","a")
    ftest_writer = open(d_dir+"22_26.txt","a")
    while True:
        valid_x,valid_y = dataset.next_valid_vec_sequences()
        if valid_x is None:
            break        
        for x,y in zip(valid_x,valid_y):
            dstr = ""
            for r in range(rnum):
                dstr+=str(dataset.spr_act_map(y[r],r))+" "
            dstr+="----"
            for f in x:
                dstr+= " " + str(f)
            fvalid_writer.write(dstr+"\n")
            
        
        fvalid_writer.write('\n')

    fvalid_writer.close()

    while True:
        test_x,test_y = dataset.next_test_vec_sequences()
        if test_x is None:
            break        
        for x,y in zip(test_x,test_y):
            dstr = ""
            for r in range(rnum):
                dstr+=str(dataset.spr_act_map(y[r],r))+" "
            dstr+="----"
            for f in x:
                dstr+= " " + str(f)
            ftest_writer.write(dstr+"\n")
            
        
        ftest_writer.write('\n')

    ftest_writer.close()
    
if __name__=="__main__":
    main()

