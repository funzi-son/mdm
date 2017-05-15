"""
Analyse CASAS ADLMR data
Son N. Tran
"""
DATA_DIR = "/home/tra161/WORK/Data/CASAS/adlmr/"

import glob
import numpy as np
import matplotlib.pyplot as plt

def get_sensor_value_range():
    files = glob.glob(DATA_DIR+"*.txt")
    dict = {}
    acts = []
    sensors_state = [];
    for f in files:
        with open(f) as f_reader:
            for line in f_reader:
                if not line:
                    continue
                strs = line.split()
                # Extract sensor states
                if strs[3]=='OF':
                    print(line + " " + f)
                if strs[2] in dict:
                    if strs[3] not in dict[strs[2]]:
                        dict[strs[2]].append(strs[3])
                else:
                    dict[strs[2]] = [strs[3]]
                # Extract activities states
                activity = [0,0]
                if len(strs)>=6:
                    activity[int(strs[4])-1] = int(strs[5])
                if len(strs)==8:
                    activity[int(strs[6])-1] = int(strs[7])
                if activity not in acts:    
                    acts.append(activity)
                # Get sensor states
                sstate = strs[2]+strs[3]
                if sstate not in sensors_state:
                    sensors_state.append(sstate)
                    
    #print(len(dict))
    #for k in dict:
    #    print((k,dict[k]))
    #print(len(acts))
    #print(acts)
    print(sensors_state)
    print(len(sensors_state))

def get_number_activities():
    files = glob.glob(DATA_DIR+"*.txt")
    activities = []
    for f in files:
        with open(f) as f_reader:
            for line in f_reader:
                if not line:
                    continue
                strs = line.split()
                if len(strs)>=6:
                    act = int(strs[5])

                if len(strs)==8:
                    act = int(strs[7])

                if act not in activities:
                    activities.append(act)
    activities.sort()
    print(activities)
                

def analyse_acts():
    files = glob.glob(DATA_DIR+"*.txt")
    matrix = np.zeros((16,16),dtype=np.float)
    for f in files:
        with open(f) as f_reader:
            for line in f_reader:
                if not line:
                    continue
                strs = line.split()
                # Extract activities states
                activity = [0,0]
                if len(strs)>=6:
                    activity[int(strs[4])-1] = int(strs[5])
                if len(strs)==8:
                    activity[int(strs[6])-1] = int(strs[7])

                matrix[activity[0],activity[1]] +=1

    mx = np.max(matrix)
    mn = np.min(matrix[np.nonzero(matrix)])
    matrix[np.nonzero(matrix)] =  (matrix[np.nonzero(matrix)] - mn)*0.8/(mx-mn) + 0.2
    
    plt.imshow(matrix,interpolation='nearest')
    plt.plot([-0.5,15.5],[-0.5,15.5])
    plt.colorbar()
    plt.xlim([-0.5,15.5])
    plt.ylim([-0.5,15.5])
    plt.tight_layout()
    plt.ylabel('Resident 1')
    plt.xlabel('Resident 2')
    plt.show()

                    
    
def main():
    #get_sensor_value_range()
    #get_number_activities()
    analyse_acts()
    
if __name__=="__main__":
    main()
