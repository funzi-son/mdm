import numpy as np
import pickle
import gzip

DAT_DIR = '/home/tra161/WORK/Data/CONLL_2000_sharedtask'
TRAIN_FILE = DAT_DIR+'/train.txt'
TEST_FILE  = DAT_DIR+'/test.txt'

# Features: <w-3> <w-2> <w-1> <w> <w+1> <w+2> <w+3> <first_name> <last_name> <capital>
def get_data_infor():
    train_size = 0
    test_size = 0
    words = []
    w_count = []
    POS = []
    NP  = []

    # CHECK TRAIN FILE
    freader = open(TRAIN_FILE,'rb')
    reading = True
    while reading:
        line = freader.readline()

        if not line:
            reading = False
        strs = [byte.decode("ascii") for byte in line.split()]
        if not strs:
            train_size+=1
            if train_size%100==0:
                print(train_size)
            continue

        if strs[0] not in words:
            words.append(strs[0])
            w_count.append(1)
        else:
            w_count[words.index(strs[0])]+=1

        if strs[1] not in POS:
            POS.append(strs[1])
        if strs[2] not in NP:
            NP.append(strs[2])


    print(train_size)
    print(len(words))
    print(len(POS))
    print(len(NP))
    print(POS)
    print(NP)
    np.savetxt(DAT_DIR+'words.csv',words,fmt='%s')
    np.savetxt(DAT_DIR+'pos.csv',POS,fmt='%s')
    np.savetxt(DAT_DIR+'np.csv',NP,fmt='%s')
    # CHECK TEST FILE
    

def crf_data_partition():

    
def generate_crf_features(dat_file):
    # Load dictionaries
    words = np.loadtxt(DAT_DIR+'words.csv',words,fmt='%s')
    pos   = np.loadtxt(DAT_DIR+'pos.csv',POS,fmt='%s')
    np    = np.loadtxt(DAT_DIR+'np.csv',NP,fmt='%s')

    freader = open(dat_file,'rb')
    reading = True
    data_sample = []
    data = []
    while reading:
        line = freader.readline()
        if not line:
            reading = False
        else:
            strs = [byte.decode("ascii") for byte in line.split()]
            w_3 =
            w__2 =
            
            if not strs:
                data.append(data_sample)
                data_sample = []
                continue
                
if __name__=="__main__":
    get_data_infor()

    
