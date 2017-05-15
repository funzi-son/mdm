'''
The Recurrent Neural Networks
Son N. Tran
sontn.fz@gmail.com
'''

import numpy as np
import tensorflow as tf
from models.sparse_rnn_cell import BasicSRNNCell

class RNN(object):
    def __init__(self,conf,dataset,model_type='rnn'):
        self.conf = conf
        self.model_type = model_type
        self.dataset = dataset
        self.max_len = dataset.get_max_len()
        self.ckp_name = conf.ckp_file

    def build_model(self):
        hidNum = self.conf.hidNum
        visNum = self.dataset.sensor_num()
        labNum = self.dataset.total_combined_acts()

        inp_ftr_len = self.dataset.inp_ftr_len() if self.conf.sparse_input else visNum
        dtype = tf.int32 if self.conf.sparse_input else tf.float32
        
        self.x = tf.placeholder(dtype,[None,self.max_len,inp_ftr_len])
        self.y = tf.placeholder(tf.float32,[None,self.max_len,labNum])
        mask  = tf.sign(tf.reduce_max(tf.abs(self.y),reduction_indices=2))
        lens = length(self.x)

        if self.conf.sparse_input:
            s,_ = tf.nn.dynamic_rnn(
                BasicSRNNCell(hidNum,visNum),
                self.x,
                dtype=tf.float32,
                sequence_length=lens
            )
        else:    
            s,_ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.GRUCell(hidNum),
                self.x,
                dtype=tf.float32,
                sequence_length=lens
            )

        s = tf.reshape(s,[-1,hidNum])
        
        with tf.variable_scope("softmax_layer"):
            weights = tf.get_variable("softmax_w",[hidNum,labNum],initializer=tf.truncated_normal_initializer(stddev=1e-1))
            biases  = tf.get_variable("softmax_b",[labNum],initializer=tf.constant_initializer(0.0))
            
        o = tf.matmul(s,weights)+biases
        o = tf.reshape(o,[-1,self.max_len,labNum])
        pred = tf.argmax(o,2)
        
        #cost = cross_entropy(o,self.y,lens,mask)

        cost = cross_entropy_with_logits(o,self.y,lens,mask)
            
        acc = accuracy(pred,tf.argmax(self.y,2),lens,mask)

#        l2 =  self.conf.weight_decay*sum(tf.nn.l2_loss(tf_var)
#                                            for tf_var in tf.trainable_variables()
#                                            if not ("Bias" in tf_var.name or "softmax_b" in tf_var.name))
#        cost += l2

        return acc,cost,pred
    
    def run(self):
        with tf.Graph().as_default():            
            lr  = self.conf.lr
            acc,loss,pred = self.build_model()
            optimizer = tf.train.GradientDescentOptimizer(lr)
            
            #tvars   = tf.trainable_variables()
            #grads,_ = tf.clip_by_global_norm(tf.gradients(loss,tvars),10)
            # Without clipping
            #grads =tf.gradients(loss,tvars) 
            #train_op = optimizer.apply_gradients(zip(grads,tvars))
            train_op = optimizer.minimize(loss)
            
            init = tf.global_variables_initializer()
            
            saver = tf.train.Saver()
            session = tf.Session()
            session.run(init)
            ######## Training
            epoch = total_err = max_vld_acc= 0
            per_dec_count = lr_decay_count = 0
            running = True
            while running:
                if not self.dataset.eof:
                    batch_x,batch_y = self.dataset.next_seq_vec_batch(batch_size=self.conf.batch_size)
                    _,err = session.run([train_op,loss],{self.x:batch_x,self.y:batch_y})
                    total_err += err
                    ### Now set the first row of weights to be zeros
                else:
                    # print('Total err %.5f' % (total_err))
                    # input('')
                    epoch +=1
                    # Evaluate on training set --- just for debugging
                    #if epoch%1 ==0:
                    #    all_trn_x,all_trn_y = self.dataset.next_seq_vec_batch()
                    #    trn_acc = session.run([acc],{self.x:all_trn_x,self.y:all_trn_y})
                    #    trn_pred = session.run([pred],{self.x:all_trn_x,self.y:all_trn_y})
                    #    accs,prec,fscore,recall = evaluation(trn_pred,all_trn_y) 
                    #    print("[Epoch %d] acc %.5f" % (epoch,trn_acc))
                    self.dataset.rewind()    
                    total_err = 0
                    # Evaluate on evaluation set
                    vld_x,vld_y = self.dataset.valid_seq_vec_dat()
                    [vld_acc] = session.run([acc],{self.x:vld_x,self.y:vld_y})
                    
                    #acc,_,_,_ = evaluation(vld_acc,vld_y)
                    #EARLY STOPPING
                    if vld_acc>=max_vld_acc:
                        max_vld_acc = vld_acc
                        saver.save(session,self.ckp_name)                    
                    else:
                        per_dec_count +=1
                        if per_dec_count >= self.conf.NUM_DEC_4_LR_DECAY:
                            saver.restore(session,self.ckp_name)
                            per_dec_count = 0
                            lr_decay_count+=1
                            lr = lr/(1+self.conf.LR_DECAY_VAL)
                    
                    print('[Epoch %d] %.5f %.5f %.5f %d %d' % (epoch,lr,vld_acc,max_vld_acc,per_dec_count,lr_decay_count))
                    if lr_decay_count >= self.conf.MAX_LR_DECAY or epoch >= self.conf.MAX_ITER:
                        running = False

            ##### Testing if exists
            tst_x,tst_y = self.dataset.test_seq_vec_dat()
            if tst_x is not None:
                saver.restore(session,self.ckp_name)
                [tst_acc] = session.run([acc],{self.x:tst_x,self.y:tst_y})
                
        
            return max_vld_acc,tst_acc

def length(x):
    # Lengths of each sequences
    mask = tf.sign(tf.reduce_max(tf.abs(x),reduction_indices=2)) 
    lens = tf.cast(tf.reduce_sum(mask,reduction_indices=1),tf.int32)
    return lens

def accuracy(o,y,length,mask):
    corrects = tf.equal(o,y)
    corrects = tf.cast(corrects,tf.float32)
    corrects*=mask

    corrects = tf.reduce_sum(corrects,reduction_indices=1)
    corrects/= tf.cast(length,tf.float32)
    return tf.reduce_mean(corrects)
    
def cross_entropy(o,y,length,mask):
    o = tf.nn.softmax(o)
    x_entr = y*tf.log(o+1e-10)
    x_entr = -tf.reduce_sum(x_entr,reduction_indices=2)
    x_entr*=mask

    x_entr = tf.reduce_sum(x_entr,reduction_indices=1)
    x_entr /= tf.cast(length,tf.float32)
    return tf.reduce_mean(x_entr)

def cross_entropy_with_logits(o,y,length,mask):
    x_entr = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=o)
    x_entr*=mask

    x_entr = tf.reduce_sum(x_entr,reduction_indices=1)
    x_entr /= tf.cast(length,tf.float32)
    return tf.reduce_mean(x_entr)
