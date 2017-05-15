import glob
import numpy as np
import matplotlib.pyplot as plt
import itertools


from numpy import genfromtxt
from models.utils import *
import os

RS_DIR = "/home/tra161/WORK/experiments/multiresidential/mdm_da/LOO/dis/CASAS_ADLMR/"
#RS_DIR = "/home/tra161/WORK/experiments/multiresidential/model_select/mdm/dis/ARAS_HouseB_10.10.10/"

def interactive_result():
    files = glob.glob(RS_DIR+"fold*_labs.csv")
    f_num = len(files)
    acc = []
    for f in files:
        labs = genfromtxt(f,delimiter=",")
        pred = genfromtxt(f.replace("labs","pred"),delimiter=",")
        acc.append(pred_accuracy(pred,labs))

        for i in range(labs.shape[0]):
            if labs[i,0] == labs[i,1]:
                print((labs[i,0],pred[i,0],pred[i,1]))
            else:
                print('TODO')
                
        print(np.mean(acc,axis=0))



def confusion_mtx_result_loo():
    files = glob.glob(RS_DIR+"fold*_labs.csv")
    f_num = len(files)

    all_labs = []
    all_pred = []
    for f in files:
        labs = genfromtxt(f,delimiter=",")
        pred = genfromtxt(f.replace("labs","pred"),delimiter=",")

        all_labs.extend(labs)
        all_pred.extend(pred)

    all_labs = np.array(all_labs)
    all_pred = np.array(all_pred)
    all_labs
    A_NUM = np.amax(all_labs)+1
    print(A_NUM)
    print(all_labs.shape)
    mtx_correct = np.zeros((A_NUM,A_NUM),dtype=np.float)
    mtx_all = np.zeros((A_NUM,A_NUM),dtype=np.float)
    for i in range(all_labs.shape[0]):
        a1 = int(all_labs[i,0])
        a2 = int(all_labs[i,1])
        mtx_all[a1,a2] +=1
        if a1 == all_pred[i,0] and a2 == all_pred[i,1]:
            mtx_correct[a1,a2] +=1
    print(mtx_all)


    mtx = mtx_correct/mtx_all
    plt.figure()
    #plot_confusion_matrix(matrix,title='a')
    plt.imshow(mtx,interpolation='nearest')
    plt.plot([-0.5,15.5],[-0.5,15.5])
    plt.colorbar()
    plt.xlim([-0.5,15.5])
    plt.ylim([-0.5,15.5])
    plt.tight_layout()
    plt.ylabel('Resident 1')
    plt.xlabel('Resident 2')
    plt.show()
    
        
def get_best_results_loo():
    max_acc = 0
    for filename in glob.iglob(RS_DIR + '/**/log.csv', recursive=True):
        rs = genfromtxt(filename,delimiter=",")
        if rs[-1,-1]>max_acc:
            max_acc = rs[-1,-1]
            max_rs = rs[-1,:]
            max_file_name = filename
    print(max_rs)
    print(max_file_name)

    
def get_best_results_ms():
    mx_evl_acc = 0
    for filename in glob.iglob(RS_DIR + '/**/log.csv', recursive=True):
        rs = genfromtxt(filename,delimiter=",")
        if rs[0,-1]>mx_evl_acc:
            mx_evl_acc = rs[0,-1]
            acc = rs[1,-1]
            print((mx_evl_acc,acc))
            max_file_name = filename
            
    print((mx_evl_acc,acc))
    print(max_file_name)

DATA = 'CASAS'
ARAS_PARTITION = '10.10.10'
def model_select_results():
    all_rs = np.zeros((3,8),np.float)
    for i in range(3):
        if i==0:
        # CASAS
            phmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_24.1.1/phmm/dis/CASAS_ADLMR/log.csv'
            chmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_24.1.1/chmm/dis/CASAS_ADLMR/log.csv'
            gd_chmm_rs  = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_24.1.1/xhmm/dis/CASAS_ADLMR/log.csv'
            fhmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_24.1.1/fhmm/dis/CASAS_ADLMR/log.csv'
            cd_fhmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_24.1.1/cd-fhmm/dis/CASAS_ADLMR/log.csv'
            gd_hmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_24.1.1/hmm/dis/CASAS_ADLMR/log.csv'

            md_hmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/mdm/dis/CASAS_ADLMR_24.1.1/1_1_1/log.csv'
            mdm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/mdm/dis/CASAS_ADLMR_24.1.1/1_0.3_0/log.csv'
            
        elif i==1:
            #ARAS A
            phmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_10.8.8_ARAS_'+ ARAS_PARTITION + '/phmm/vec1/ARAS_HouseA/log.csv'
            chmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_10.8.8_ARAS_'+ ARAS_PARTITION + '/chmm/vec1/ARAS_HouseA/log.csv'
            gd_chmm_rs  = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_10.8.8_ARAS_'+ ARAS_PARTITION + '/xhmm/vec1/ARAS_HouseA/log.csv'
            fhmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_10.8.8_ARAS_'+ ARAS_PARTITION + '/fhmm/vec1/ARAS_HouseA/log.csv'
            cd_fhmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_10.8.8_ARAS_'+ ARAS_PARTITION + '/cd-fhmm/vec1/ARAS_HouseA/log.csv'
            gd_hmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_10.8.8_ARAS_'+ ARAS_PARTITION + '/hmm/vec1/ARAS_HouseA/log.csv'

            md_hmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/mdm/vec1/ARAS_HouseA_'+ ARAS_PARTITION + '/1_1_1/log.csv'
            #mdm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/mdm/vec1/ARAS_HouseA/1_1_0.9/log.csv'
            mdm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/mdm/dis/ARAS_HouseA_'+ ARAS_PARTITION + '/6_6_0.1/log.csv'
        else:
            #ARAS B
            phmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_10.8.8_ARAS_'+ ARAS_PARTITION + '/phmm/dis/ARAS_HouseB/log.csv'
            chmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_10.8.8_ARAS_'+ ARAS_PARTITION + '/chmm/dis/ARAS_HouseB/log.csv'
            gd_chmm_rs  = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_10.8.8_ARAS_'+ ARAS_PARTITION + '/xhmm/dis/ARAS_HouseB/log.csv'
            fhmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_10.8.8_ARAS_'+ ARAS_PARTITION + '/fhmm/dis/ARAS_HouseB/log.csv'
            cd_fhmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_10.8.8_ARAS_'+ ARAS_PARTITION + '/cd-fhmm/dis/ARAS_HouseB/log.csv'
            gd_hmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/CASAS_10.8.8_ARAS_'+ ARAS_PARTITION + '/hmm/dis/ARAS_HouseB/log.csv'

            md_hmm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/mdm/dis/ARAS_HouseB_'+ ARAS_PARTITION + '/1_1_1/log.csv'
            mdm_rs = '/home/tra161/WORK/experiments/multiresidential/model_select/mdm/dis/ARAS_HouseB_'+ ARAS_PARTITION + '/10_100_5/log.csv'
            
        rs = genfromtxt(phmm_rs,delimiter=",")
        all_rs[i,0] = rs[-1,-1]
        rs = genfromtxt(chmm_rs,delimiter=",")
        all_rs[i,1] = rs[-1,-1]
        rs = genfromtxt(gd_chmm_rs,delimiter=",")
        all_rs[i,2] = rs[-1,-1]
        rs = genfromtxt(fhmm_rs,delimiter=",")
        all_rs[i,3] = rs[-1,-1]
        rs = genfromtxt(cd_fhmm_rs,delimiter=",")
        all_rs[i,4] = rs[-1,-1]
        rs = genfromtxt(gd_hmm_rs,delimiter=",")
        all_rs[i,5] = rs[-1,-1]

        if md_hmm_rs is not None:
            rs = genfromtxt(md_hmm_rs,delimiter=",")
            all_rs[i,6] = rs[-1,-1]
        if mdm_rs is not None:
            rs = genfromtxt(mdm_rs,delimiter=",")
            all_rs[i,7] = rs[-1,-1]
        
    print(all_rs)
    width = 0.1
    ind = np.arange(3)
    fig,ax = plt.subplots()
    rects = []
    colors = ['#17B4E8','#1729E8','#045706','#870AC2','#05965A','#5A5E06','#C2810A','#C20A0A']
    for i in range(8):
        rects.append(ax.bar(ind + width*i,all_rs[:,i]*100,width,color=colors[i]))

    print(all_rs[:,-1] - np.amax(all_rs[:,0:6],axis=1))
    '''
    ax.set_ylabel('Accuracy')
    ax.set_xticks(ind + 3*width)
    ax.set_xticklabels(('CASAS','ARAS House A','ARAS House B'))
    ax.legend([r[0] for r in rects],('pHMM','cHMM','gd-cHMM','fHMM','cd-fHMM','gd-HMM','md-HMM','MDM'),loc=0,ncol=2)
    plt.ylim([10,76])
    plt.savefig('/home/tra161/WORK/projects/multiresidential/figs/result.png')
    plt.show()
    '''
def hmm_vs_dectree():
    house = 'ARAS_HouseA'
    EXP_DIR = '/home/tra161/WORK/experiments/multiresidential/DecTreePartition/'
    if house=='ARAS_HouseA':
        dectree_rs = [43.56,48.36,48.53,49.28]
    else:
        dectree_rs = [30.47,64.19,81.08,84.45]

    mdm_rs = []
    for d in ['1_1','1_7','1_14','1_21']:
        max_acc = 0
        for filename in glob.iglob(EXP_DIR + d +'/model_select/mdm/dis/' + house+  '/**/log.csv', recursive=True):
            rs = genfromtxt(filename,delimiter=",")
            if rs[0,-1]>max_acc:
                max_acc = rs[0,-1]
                max_rs = rs[-1,-1]*100
                max_file_name = filename
                
        mdm_rs.append(max_rs)
    print([a-b for a,b in zip(mdm_rs,dectree_rs)])
    print(dectree_rs)
    
    #fig,ax = plt.figure(figsize=(4, 5), dpi=100)
    fig,ax = plt.subplots(figsize=(4, 5), dpi=100)
    ax.plot([1,7,14,21],mdm_rs,'r-s',label='MDM')
    ax.plot([1,7,14,21],dectree_rs,'g-^',label='IDT')
    plt.xlim([0,23])
    if house=='ARAS_HouseA':
        plt.ylim([25,90])
    else:
        plt.ylim([25,90])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of days for training')
    ax.legend(loc=4,shadow=True)
    plt.savefig('/home/tra161/WORK/projects/multiresidential/figs/mdm_vs_idt_'+house+'_.png')
    plt.show()
    
def main():
    #interactive_result()
    #get_best_results_ms()
    get_best_results_loo()
    #confusion_mtx_result_loo()
    #model_select_results()
    #hmm_vs_dectree()
    
if __name__=="__main__":
    main()
