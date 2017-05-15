import numpy as np

def save_result(preds,labs,save_dir):
    np.savetxt(save_dir+'_pred.csv',preds,delimiter=',')
    np.savetxt(save_dir+'_labs.csv',labs,delimiter=',')
    
def pred_accuracy(preds,labs):
    rnum = len(preds[0]) 
    acc = (rnum+1)*[0]
    for i in range(len(preds)):
        all_match = True
        for j in range(rnum):
            if preds[i][j] == labs[i][j]:
                acc[j]+=1
            else:
                all_match = False
        if all_match: # All match
            acc[rnum] +=1
    
    return [x/len(preds) for x in acc]


def evaluation(preds,labs): ## TODO
    rnum = len(preds[0]) 
    acc = (rnum+1)*[0]
    for i in range(len(preds)):
        all_match = True
        for j in range(rnum):
            if preds[i][j] == labs[i][j]:
                acc[j]+=1
            else:
                all_match = False
        if all_match: # All match
            acc[rnum] +=1
    
    return [x/len(preds) for x in acc]


'''
def evaluation(preds,labs,labnums):
    
    rnum = len(preds)
    sNum = len(preds[0])
    labnums = [x.shape[2] for x in preds]
    
    tp = np.array([0]*np.prod(labnums))
    fp = np.array([0]*np.prod(labnums))
    fn = np.array([0]*np.prod(labnums))
    mask_ = tf.sign(tf.reduce_max(tf.abs(labs), reduction_indices=2))
    lens = tf.cast(tf.reduce_sum(mask,reduction_indices=1),tf.int32)
    
    accs = [0]*(rnum+1)
    for i in range(len(preds)):
        preds[i]   = np.argmax(preds[i],axis=2)
        labs [i]   = np.argmax(labs[i],axis=2)

    for s in range(snum):
        for t in range(lens[s]):
            pred_inx = 0
            targ_inx = 0
            for r in range(rnum):
                pred_inx+= 
                targ_inx+= 
                if preds[r][s,t] == labs[r][s,t]:
                    accs[r]+=1
                
                    
            if pred_inx==targ_inx:
                accs[r+1] +=1
                tp[targ_inx] +=1 # CHECK
            else:
                fp[targ_inx] +=1 # CHECK
                fn[pred_inx] +=1 # CHECK

    # Compute precision & f-score
    tp_av = fp_av = fn_av = 0
    for i in range(len(tps)):
        tp_av += tp[i]
        fp_av += fp[i]
        fn_av += fn[i]

    precision = tp_av*1.0/(tp_av + fp_av)
    recall    = tp_av*1.0/(tp_av + fn_av)
    fscore    = 2.0*precision*recall/(precision+recall)
    
    accs =[acc/np.sum(lens) for acc in accs]
    return accs,precision,fscore,recall


'''
