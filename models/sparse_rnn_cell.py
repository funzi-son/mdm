from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import multiply
from tensorflow.python.framework.constant_op import constant

import tensorflow as tf

class BasicSRNNCell(RNNCell):
    """ Basic RNNCell with sparse input """
    def __init__(self,num_units,input_size,activation=tanh,reuse=None):
        if input_size is None:
            raise ValueError("input size is needed for sparse computation")
        self._input_size = input_size
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse
        self.p = constant([0.0]+[1.0]*input_size,shape=[1,input_size+1])

    @property
    def state_size(self):
        return self._num_units
    
    @property
    def output_size(self):
        return self._num_units

    def __call__(self,inputs,state,scope=None):
        with vs.variable_scope(scope or "basic_sparse_rnn_cell"):
            output = self._activation(_sparse_linear(inputs,state,self._input_size,self._num_units,self.p)) 
        return output,output
    
def _sparse_linear(inp,state,input_size,output_size,p): 
    scope = vs.get_variable_scope()
    ftr_len = inp.get_shape()[0] # CHECK THIS
    dtype = inp.dtype
    with vs.variable_scope(scope) as outer_scope:
        Wvh = vs.get_variable("wvh",[input_size+1,output_size],dtype=tf.float32)
        Wvh = Wvh # This may cause computational overhead
        # The first row (all zeros) is for feature padding (not sequence padding)
        
        Whh = vs.get_variable("whh",[output_size,output_size],dtype=tf.float32)
        
        res = tf.matmul(state,Whh) + tf.reduce_sum(tf.nn.embedding_lookup(Wvh,inp),axis=1)
        
        with vs.variable_scope(outer_scope):
            bias = vs.get_variable("bias",[output_size],dtype=tf.float32,initializer=init_ops.constant_initializer(0,dtype=tf.float32))
            res  = nn_ops.bias_add(res,bias)
        
    return res
