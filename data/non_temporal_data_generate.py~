'''
Convert Aras Data into non-temporal format: x = sensors' state from t-L to t, y = activities at time t
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

L = 3
data_id = 0
def main():

    FOLD_NUM = fold_num[data_id]
    DATA_PATH = path[data_id]
    d_dir = D_DIR[data_id] + '/non_temporal/clength'+ str(L)+ '/'
    if not os.path.isdir(d_dir):
        os.makedirs(d_dir)

            
    for fold in range(FOLD_NUM):
        dataset = data[data_id](DATA_PATH,fold+1)
                
        sensor_size = dataset.sensor_num()
        input_dim  = L*sensor_size
        train_data  = np.empty((0,L*sensor_size),dtype=float)
        train_label = np.empty((0,0),dtype=float)
        valid_data  = np.empty((0,L*sensor_size),dtype=float)
        valid_label = np.empty((0,0),dtype=float)

        # Get train data
        d = np.zeros((1,input_dim),dtype=float)
        
        while True:
            prev_act,curr_act,sensor = dataset.next()
            if prev_act is None and curr_act is None and sensor is None:
                break
            if data_id ==0: #CASAS
                dataset.set_vector_type(3)
            elif data_id ==1: # ARAS House A
                dataset.set_vector_type(1) 
            else:   # ARAS House B
                dataset.set_vector_type(1)

                
            sensor = dataset.sensor_vec(sensor)
            d = np.roll(d,-sensor_size)
            d[0,-sensor_size:] = sensor
            train_data = np.append(train_data,d,axis=0)

            curr_act = dataset.act_map(curr_act)
            train_label = np.append(train_label,curr_act)

            
        # Get evaluation data
        d = np.zeros((1,input_dim),dtype=float)
        valid_x,valid_y = dataset.valid_vec_sequences()
        for x,y in zip(valid_x,valid_y):
            d = np.roll(d,-sensor_size)
            d[0,-sensor_size:] = x
            valid_data = np.append(valid_data,d,axis=0)

            l = dataset.act_map(y)
            valid_label = np.append(valid_label,l)


        np.savetxt(d_dir + 'fold'+str(fold+1)+'_train_data.csv',train_data)
        np.savetxt(d_dir + 'fold'+str(fold+1)+'_train_label.csv',train_label)
        np.savetxt(d_dir + 'fold'+str(fold+1)+'_valid_data.csv',valid_data)
        np.savetxt(d_dir + 'fold'+str(fold+1)+'_valid_label.csv',valid_label)

        print(train_data.shape)
        print(train_label.shape)
        print(valid_data.shape)
        print(valid_label.shape)
if __name__=="__main__":
    main()
