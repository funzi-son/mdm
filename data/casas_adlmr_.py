"""
Casas adlmr reader
Son N. Tran
"""

import numpy as np

from collections import OrderedDict

SENSORS_DICT = OrderedDict([('M01', ['ON', 'OFF']),
                            ('M02', ['ON', 'OFF']),
                            ('M03', ['ON', 'OFF']),
                            ('M04', ['ON', 'OFF']),
                            ('M05', ['ON', 'OFF']),
                            ('M06', ['ON', 'OFF']),
                            ('M07', ['ON', 'OFF']),
                            ('M08', ['ON', 'OFF']),
                            ('M09', ['ON', 'OFF']),
                            ('M10', ['ON', 'OFF']),
                            ('M11', ['ON', 'OFF']),
                            ('M12', ['ON', 'OFF']),
                            ('M13', ['ON', 'OFF']),
                            ('M14', ['ON', 'OFF']),
                            ('M15', ['ON', 'OFF']),
                            ('M16', ['ON', 'OFF']),
                            ('M17', ['ON', 'OFF']),
                            ('M18', ['ON', 'OFF']),
                            ('M19', ['ON', 'OFF']),
                            ('M20', ['ON', 'OFF']),
                            ('M21', ['ON', 'OFF']),
                            ('M22', ['ON', 'OFF']),
                            ('M23', ['ON', 'OFF']),
                            ('M24', ['ON', 'OFF']),
                            ('M25', ['ON', 'OFF']),
                            ('M26', ['ON', 'OFF']),
                            ('M51', ['ON', 'OFF']),
                            ('I04', ['PRESENT','ABSENT']),
                            ('I06', ['PRESENT','ABSENT']),
                            ('D07', ['OPEN', 'CLOSE']),
                            ('D09', ['OPEN', 'CLOSE']),
                            ('D10', ['OPEN', 'CLOSE']),
                            ('D11', ['OPEN', 'CLOSE']),
                            ('D12', ['OPEN', 'CLOSE']),
                            ('D13', ['OPEN', 'CLOSE']),
                            ('D14', ['OPEN', 'CLOSE']),
                            ('D15', ['OPEN', 'CLOSE'])])
FOLD_NUM = 26
SENSOR_NUM = 37
RESIDENT_NUM = 2
SENSOR_VALUES = {'ON':1, 'OFF':0,
                 'OPEN':1, 'CLOSE':0,
                 'PRESENT':1,'ABSENT':0}
ACT_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

ACT_DICT = [[1, 0],  [0, 2],  [4, 0],  [0, 3],  [0, 5],
           [6, 0],  [0, 7],  [6, 7],  [9, 0],  [0, 8],
           [10, 0], [0, 11], [0, 13], [12, 0], [13, 13],
           [0, 15], [14, 0], [14, 15],[0, 0],  [9, 8],
           [10, 11],[10, 1], [13, 0], [12, 13],[15, 0],
           [0, 4],  [10, 15]
]

SPR_ACTS = [[ 0,  1,  4,  6,  9, 10, 12, 13, 14, 15],[ 0,  1,  2,  3,  4,  5,  7,  8, 11, 13, 15]]

class CASAS_ADLMR(object):
    
    def __init__(self,data_path,fold_id=-1):
        if fold_id>0:
            train_fold_ids =  list(set(range(FOLD_NUM+1)[1:])-set([fold_id]))
            self.train_files = [data_path+"P%.2d.txt" % i for i in train_fold_ids]
            self.valid_file = data_path+"P%.2d.txt" % fold_id
            self.all_files = self.train_files+[self.valid_file]
        else:
            self.train_files = [data_path+"P%.2d.txt" % i for i in range(1,17)]
            self.valid_files = [data_path+"P%.2d.txt" % i for i in range(17,22)]
            self.test_files  = [data_path+"P%.2d.txt" % i for i in range(22,27)]
            self.all_files = self.train_files+self.valid_files+self.test_files

        self.fcount = 0
        self.file_inx = 0
        self.train_f_num = len(self.train_files)
        self.freader = None
        self.prior = np.zeros((self.total_separate_acts(),2),dtype=np.float)
        self.spr_prior = []       
        for i in range(RESIDENT_NUM):
            self.spr_prior.append(np.zeros((1,len(SPR_ACTS[i])),dtype=np.float))
            
        self.cmb_prior = np.zeros((1,self.total_combined_acts()),dtype=np.float)
        self.sensor_state = np.ones((SENSOR_NUM,),dtype=np.int)

        # This is for batch learning
        self.eof = False
        self.start_inx = -1
        self.end_inx = -1

        # This is for recurrent nets
        self.max_len = 0
        
    def next(self):
        # trans: trans[i,actor] -> trans[i+1,actor]
        # emits: emits[sensor_value,actor1,actor2]
        curr_act = None
        sensor   = None
        NODATA = True
        first_acts = False
        while NODATA:
            if self.freader is not None:
                line = self.freader.readline()
                
                if not line:
                    self.fcount+=1
                    self.freader = None
            
            if self.fcount<self.train_f_num:
                if self.freader is None:
                    file_name = self.train_files[self.fcount]
                    #print(file_name)
                    #print("read_file %s %d"%(file_name,self.train_f_num))
                    self.freader = open(file_name,"r")
                    self.eof = False
                    self.prev_act = None
                    line = self.freader.readline()
                    first_acts = True
                
                strs = line.split()
                sensor = [strs[2],strs[3]]
                #x_inx= sensor2discrete(strs[2],strs[3])
                
                # prev_act & curr_act
                prev_act = self.prev_act
                curr_act = [0,0]
                if len(strs)>4:
                    curr_act[int(strs[4])-1]=int(strs[5])
                if len(strs)== 8:
                    curr_act[int(strs[6])-1]=int(strs[7])
                    
                # Prior
                if first_acts:
                    # prior for separate labels
                    self.prior[curr_act[0],0] +=1
                    self.prior[curr_act[1],1] +=1
                    # prior for combined labels
                    self.cmb_prior[0,self.act_map(curr_act)] +=1
                    # prior for separete resident (may have different set of labels)
                    self.spr_prior[0][0,self.spr_act_map(curr_act[0],0)]+=1
                    self.spr_prior[1][0,self.spr_act_map(curr_act[1],1)]+=1
                    
                    first_acts = False                    
                # 
                self.prev_act =  curr_act
                
                NODATA  = False
            else:
                self.eof = True
                return None,None,None
            
        return prev_act,curr_act,sensor

    

    def get_prior(self,type=1):
        if type==1:
            return self.cmb_prior
        else:
            return self.spr_prior
            
    def valid_dis_sequences(self):
        return self.load_dis_sequence(self.valid_file)

    def next_valid_dis_sequences(self):
        if self.file_inx >= len(self.valid_files):
            self.file_inx = 0
            return None, None
        
        x,y = self.load_dis_sequence(self.valid_files[self.file_inx])
        self.file_inx += 1
        return x,y

    def valid_da_dis_sequences(self):
        #print(self.valid_file)
        reader_ = open(self.valid_file,"r")
        x1 = []
        x2 = []
        y = []
        while True:
            line = reader_.readline()
            if not line:
                break
            strs = line.split()
            x_ = self.sensor_map([strs[2],strs[3]])+1
            
            y_ = [0,0]
            if len(strs)>4:
                y_[int(strs[4])-1]=int(strs[5])
                
            if len(strs)== 8:
                y_[int(strs[6])-1]=int(strs[7])
            y.append(y_)

            if y_[0] ==0:
                x1.append(0)
            else:
                x1.append(x_)

            if y_[1] ==0:
                x2.append(0)
            else:
                x2.append(x_)
                
        return [x1,x2],y

    
    def next_test_dis_sequences(self):
        if self.file_inx >= len(self.valid_files):
            self.file_inx = 0
            return None, None
        
        x,y  =  self.load_dis_sequence(self.test_files[self.file_inx])
        self.file_inx+=1
        return x,y
    
    def load_dis_sequence(self,datafile):
        #print(datafile)
        reader_ =  open(datafile,"r")
        x = []
        y = []
        while True:
            line = reader_.readline()
            if not line:
                break
            strs = line.split()
            x_ = self.sensor_map([strs[2],strs[3]])
            x.append(x_)
            y_ = [0,0]
            if len(strs)>4:
                y_[int(strs[4])-1]=int(strs[5])
                
            if len(strs)== 8:
                y_[int(strs[6])-1]=int(strs[7])
            y.append(y_)
        return x,y

    def valid_vec_sequences(self):
        return self.load_vec_sequence(self.valid_file)

    def next_valid_vec_sequences(self):
        if self.file_inx >= len(self.valid_files):
            self.file_inx = 0
            return None, None
        
        x,y = self.load_vec_sequence(self.valid_files[self.file_inx])
        self.file_inx += 1
        return x,y

    def next_test_vec_sequences(self):
        if self.file_inx >= len(self.valid_files):
            self.file_inx = 0
            return None, None
        
        x,y  =  self.load_vec_sequence(self.test_files[self.file_inx])
        self.file_inx+=1
        return x,y

    
    def load_vec_sequence(self,datafile):
        #print(datafile)
        reader_ =  open(datafile,"r")
        x = []
        y = []
        while True:
            line = reader_.readline()
            if not line:
                break
            strs = line.split()
            x_ = self.sensor_vec([strs[2],strs[3]])
            x.append(x_)
            y_ = [0,0]
            if len(strs)>4:
                y_[int(strs[4])-1]=int(strs[5])
                
            if len(strs)== 8:
                y_[int(strs[6])-1]=int(strs[7])
            y.append(y_)
        return x,y
    
    
    def total_sensor_values(self):
        tvals = 0
        for i in SENSORS_DICT.values():
            tvals = tvals+len(i)
        return tvals

    def sensor_num(self):
        return len(SENSORS_DICT)
    
    def resident_num(self):
        return RESIDENT_NUM
    
    def total_separate_acts(self):
        # Number of unique activities by person
        return len(ACT_VALUES) + 1 # 0 means not being seen doing anything

    def separate_act_nums(self):
        return [len(SPR_ACTS[0]),len(SPR_ACTS[1])]
    
    def total_combined_acts(self):
        #Number of all possible activities
        return len(ACT_DICT)
    
    def sensor_map(self,sensor):
        sensor_id = sensor[0]
        sensor_val = sensor[1]
        # Convert  sensor value to a discrete (started from 0)
        dis_val = 0
        for i in SENSORS_DICT:
            if i==sensor_id:
                dis_val += SENSORS_DICT[sensor_id].index(sensor_val)
                return dis_val
            dis_val += len(SENSORS_DICT[i])
            
    def sensor_vec(self,sensor):
        sensor_id = sensor[0]
        sensor_val = sensor[1]
        # Convert sensor value to a binary vector
        sensor_inx = list(SENSORS_DICT).index(sensor_id)
        sensor_ival = SENSORS_DICT[sensor_id].index(sensor_val)
        if self.vec_type==1: # state of all sensors
            self.sensor_state[sensor_inx] = sensor_ival
            sensor = self.sensor_state.tolist()
        elif self.vec_type==2: # The sensors change their states are set to 1, others = 0
            sensor = np.zeros((len(SENSORS_DICT),),dtype=np.int)
            if self.sensor_state[sensor_inx]!=sensor_ival:
                sensor[sensor_inx] = 1
            self.sensor_state[sensor_inx] = sensor_ival
        elif self.vec_type==3: # If the sensor's state is recored -> set to 1, others = 0
            sensor = np.zeros((len(SENSORS_DICT),),dtype=np.int)
            sensor[sensor_inx] = 1
        else:
            raise ValueError('vector type is not correct!!!')       

        return sensor
    
    def set_vector_type(self,vtype):
        self.vec_type = vtype
        
    def get_act_dict(self):
        return ACT_DICT
    
    def act_map(self,act_vec):
        return ACT_DICT.index(act_vec)
    
    def act_rmap(self,act_inds):
        acts = []
        for a in act_inds:
            acts.append(ACT_DICT[a])
        return acts
    def spr_act_map(self,act_id,resident_id):
        return SPR_ACTS[resident_id].index(act_id)


    def arrange_batch(self,bsize):
        self.start_inx = self.end_inx+1
        self.end_inx = self.end_inx+bsize

        xs = []
        ys = []
        for i in range(self.start_inx,self.end_inx+1,1):
            f_inx = i
            if f_inx>self.end_inx or f_inx>len(self.train_files)-1:
                break
            else:
                x = []
                y = []
                self.freader = open(self.train_files[f_inx],"r")
                while True:
                    line = self.freader.readline()
                    if not line:
                        break
                    strs = line.split()
                    sensor = [strs[2],strs[3]]
                    act    = [0,0]
                    if len(strs)>4:
                        act[int(strs[4])-1] = int(strs[5])
                        if len(strs)==8:
                            act[int(strs[6])-1] = int(strs[7])
                            
                    x.append(self.sensor_map(sensor))
                    y.append(self.act_map(act))
                xs.append(x)
                ys.append(y)

        if self.end_inx+1 >= len(self.train_files):
            self.eof = True
            
        return xs,ys
    
    def next_seq_vec_batch(self,batch_size=0,multilabel=False):    
        #Return a batch of sequence: batchnum x max_len x xsize with zeros padding
        # Todo - multilabel
        if batch_size ==0:
            batch_size = len(self.train_files)
        
        self.start_inx = self.end_inx+1
        self.end_inx = min(self.end_inx+batch_size,len(self.train_files)-1)

        if multilabel:
            labnums = self.separate_act_num()
        else:
            labnums = [self.total_combined_acts()]
            
        xs = np.zeros((self.end_inx-self.start_inx+1,self.max_len,self.sensor_num()),dtype=np.int32)

        ys=[]
        for y_size in labnums:
            ys.append(np.zeros((xs.shape[0],self.max_len,y_size),dtype=np.int32))
            
        for i in range(xs.shape[0]):
            f_inx = i+self.start_inx
            #print(self.train_files[f_inx])
            self.freader = open(self.train_files[f_inx],"r")
            t = 0
            while True:
                line = self.freader.readline()
                if not line:
                    break
                strs = line.split()
                sensor = [strs[2],strs[3]]
                act    = [0,0]
                if len(strs)>4:
                    act[int(strs[4])-1] = int(strs[5])
                    if len(strs)==8:
                        act[int(strs[6])-1] = int(strs[7])

                x_vec = self.sensor_vec(sensor)

                if multilabel:
                    print('TODO')
                else: 
                    y_vec = [0] *labnums[0]
                    y_vec[self.act_map(act)] = 1
                    xs[i,t,:] = x_vec
                        
                    ys[0][i,t,:] = y_vec
                    t +=1
        if self.end_inx+1 >= len(self.train_files):
            self.end_inx = -1
            self.eof = True

        return xs,ys[0]

    def valid_seq_vec_dat(self,multilabel=False):    
        #Return a batch of sequence: batchnum x max_len x xsize with zeros padding
        # TODO - multilabel
        if multilabel:
            labnums = self.separate_act_num()
            files = self.valid_files
        else:
            labnums = [self.total_combined_acts()]
            files = [self.valid_file]

        batch_size = len(files)
        xs = np.zeros((batch_size,self.max_len,self.sensor_num()),dtype=np.int32)
        ys=[]
        for y_size in labnums:
            ys.append(np.zeros((batch_size,self.max_len,y_size),dtype=np.int32))

        for i in range(batch_size):
            #print(self.train_files[f_inx])
            self.freader = open(files[i],"r")
            t = 0
            while True:
                line = self.freader.readline()
                if not line:
                    break
                strs = line.split()
                sensor = [strs[2],strs[3]]
                act    = [0,0]
                if len(strs)>4:
                    act[int(strs[4])-1] = int(strs[5])
                if len(strs)==8:
                    act[int(strs[6])-1] = int(strs[7])

                x_vec = self.sensor_vec(sensor)
                xs[i,t,:] = x_vec
                
                if multilabel:
                    print('TODO')
                else: 
                    y_vec = [0] *labnums[0]
                    y_vec[self.act_map(act)] = 1
                    ys[0][i,t,:] = y_vec
                    
                t +=1

        return xs,ys[0]

    
    def get_max_len(self):
        if self.max_len==0:
            
            for file_path in self.all_files:
                self.freader = open(file_path,"r")
                seq_len = 0
                while True:
                    line = self.freader.readline()
                    if not line:
                        break
                    seq_len +=1
                if seq_len >self.max_len:
                    self.max_len = seq_len
            
    
        return self.max_len
            
    def rewind(self):
        self.start_inx = -1
        self.end_inx = -1
        self.eof = False
    
#    raise ValueError('sensor id and,or value do not exist')
         

            
if __name__ == "__main__":
    data = CASAS_ADLMR("P01")
    
