'''
Long-short term memory (based on ptb_world_lm)
Son N. Tran
'''
import time
import copy

import numpy as np

import tensorflow as tf

class LSTMGraph(object):
    def __init__(self,is_training,conf):
        self._num_steps = num_steps = conf.num_steps
        self._batch_size = batch_size = conf.batch_size
        h_num = conf.hidden_size
        v_num = conf.vocab_size
        print(batch_size)
        print(num_steps)
        self.x = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.y = tf.placeholder(tf.int32, [batch_size, num_steps])
        
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(h_num, forget_bias=0.0,state_is_tuple=True)
        if is_training and conf.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=conf.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * conf.num_layers,state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size,tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding",[v_num,h_num])
            x = tf.nn.embedding_lookup(embedding,self.x)

        if is_training and conf.keep_prob <1:
            x = tf.nn.dropout(x,conf.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for t in range(num_steps):
                if t >0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output,state) = cell(x[:,t,:],state)
                outputs.append(cell_output)
                
        output    = tf.reshape(tf.concat(1,outputs),[-1,h_num])
        softmax_w = tf.get_variable("softmax_w",[h_num,v_num]) 
        softmax_b = tf.get_variable("softmax_b",[v_num])
        logits    = tf.matmul(output,softmax_w) + softmax_b
        loss      = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.y,[-1])],
            [tf.ones([batch_size*num_steps])])
                                                            
        self._cost = cost = tf.reduce_sum(loss)/batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0,trainable=False)
        tvars = tf.trainable_variables()
        grads,_ = tf.clip_by_global_norm(tf.gradients(cost,tvars),conf.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads,tvars))
    
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self._lr, lr_value))
    
class FzLSTM(object):
    
    def __init__(self,dataset,conf,eval_conf=None):
        self._dataset = dataset
        self._conf = conf
        if eval_conf is None:
            self._eval_conf = copy.copy(self._conf)
            self._eval_conf.batch_size = 1
            self._eval_conf.num_steps = 1
        else:
            self._eval_conf = eval_conf
            
    def run(self):
        with tf.Graph().as_default(), tf.Session() as session:
            initializer = tf.random_uniform_initializer(-self._conf.init_scale,
                                                        self._conf.init_scale)
            with tf.variable_scope("model",reuse=None,initializer=initializer):
                mtrain = LSTMGraph(is_training=True,conf=self._conf)
            with tf.variable_scope("model",reuse=True,initializer=initializer):
                mvalid = LSTMGraph(is_training=False,conf=self._conf)
                mtest  = LSTMGraph(is_training=False,conf=self._eval_conf)

            tf.initialize_all_variables().run()

            for i in range(self._conf.max_epoch):
                lr_decay = self._conf.lr_decay**max(i-self._conf.max_epoch,0.0)
                mtrain.assign_lr(session,self._conf.learning_rate*lr_decay)

                print("[LSTM] Epoch: %d Learning rate: %.3f" % (i+1,session.run(mtrain._lr)))
                self._dataset.set_mode('train',mtrain._batch_size,mtrain._num_steps)
                train_eval = run_epoch(session, mtrain,self._dataset,mtrain._train_op,verbose=True)
                print("      - Train eval: %.3f" %train_eval)
                self._dataset.set_mode('validation',mvalid._batch_size,mvalid._num_steps)
                valid_eval = run_epoch(session,mvalid,self._dataset,tf.no_op())
                print("      - Valid eval: %.3f" %valid_eval)

            self._dataset.set_mode('test',mtest._batch_size,mtest._num_steps)
            test_eval = run_epoch(session,mtest,self._dataset,tf.no_op())
            print("[LSTM] Test result = %.3f" % test_eval)
        
def run_epoch(session,m,dataset,_op,verbose=False):
    state = session.run(m._initial_state)
    total_cost = 0.0
    iters = 0

    start_time = time.time()
    for step,(x,y) in enumerate(dataset.get_data()): 
        dict = {m.x:x, m.y:y,m._initial_state:state}
        cost, state,_ = session.run([m._cost,m._final_state,_op],dict)
        total_cost += cost
        iters += m._num_steps
    print("[LSTM] time: %.3f" % (time.time()-start_time))
    return np.exp(total_cost/iters)
