"""
Analyse ARAS data
Son N. Tran
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
import itertools


import operator

act_count_A = {0: 1, 1: 1, 6: 1, 11: 1, 12: 1, 15: 1, 16: 1, 18: 1, 19: 1, 26: 1, 8: 8, 24: 8, 7: 87, 21: 289, 14: 300, 22 : 421, 9 : 621, 20 : 724, 13: 2706, 17: 3388, 23: 4146, 2: 5648, 25: 5994, 10: 7201, 3: 7206, 5: 23353, 4: 28903}

act_count_B = {1: 1, 3: 1, 10: 1, 11: 1, 14: 1, 15: 1, 16: 1, 19: 1, 21: 1, 26: 1, 24: 2, 5: 7, 0: 66, 20: 215, 7: 276, 22: 439, 9: 743, 13: 822, 18: 1855, 17: 1880, 8: 1923, 23: 2648, 2: 5289, 6: 7507, 4: 10187, 12: 65581}

DATA_DIR = "/home/tra161/WORK/Data/ARAS/"
HOUSE = "House_B"

def get_infor():
    files = glob.glob(DATA_DIR + HOUSE + "/*.txt")
    acts = {}
    act_pairs = []
    sensors = []
    not_binary = False
    for f in files:
        with open(f) as f_reader:
            for line in f_reader:
                strs = line.split(" ")
                s = 0
                for i in range(20):
                    sval = int(strs[i])
                    s += sval
                    if sval !=0 and sval !=1:
                        not_binary = True
                #if s>1:
                #    print('more than 1 sensor is activated')
                a_pair = [int(strs[20])-1,int(strs[21].rstrip('\n'))-1]
                s_state = [int(x) for x in strs[0:20]]
                if a_pair not in act_pairs:
                    act_pairs.append(a_pair)
                if s_state not in sensors:
                    sensors.append(s_state)
                
                if a_pair[0] not in acts:
                    acts[a_pair[0]] =  1
                else:
                    acts[a_pair[0]] += 1
                if a_pair[1] not in acts:
                    acts[a_pair[1]] = 1
                else:
                    acts[a_pair[1]] = 1

    if not_binary:
        print('Not binary values')
    #print(act_pairs)
    print(sensors)
    #acts = sorted(acts.items(),key=operator.itemgetter(1))
    #print(acts)

def act_analysis():
    files = glob.glob(DATA_DIR + HOUSE + "/*.txt")
    matrix = np.zeros((27,27),dtype=np.float)
    for f in files:
        with open(f) as f_reader:
            for line in f_reader:
                strs = line.split(" ")

                matrix[int(strs[20])-1,int(strs[21].rstrip('\n'))-1]+=1

    mx = np.max(matrix)
    mn = np.min(matrix[np.nonzero(matrix)])
    matrix[np.nonzero(matrix)] =  (matrix[np.nonzero(matrix)] - mn)*0.8/(mx-mn) + 0.2
    #np.set_printoptions(precision=2)
    plt.figure()
    #plot_confusion_matrix(matrix,title='a')
    
    plt.imshow(matrix,interpolation='nearest')
    plt.plot([-0.5,26.5],[-0.5,26.5])
    plt.colorbar()
    plt.xlim([-0.5,26.5])
    plt.ylim([-0.5,26.5])
    plt.tight_layout()
    plt.ylabel('Resident 1')
    plt.xlabel('Resident 2')
    plt.show()

def plot_confusion_matrix(cm, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45)
    #plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i,j]>0:
            plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Resident 1')
    plt.xlabel('Resident 2')

    
if __name__=='__main__':
    #get_infor()
    act_analysis()
