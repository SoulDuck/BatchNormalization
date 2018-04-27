# Batch Normalization 

Batch Normalization [Original paper](https://arxiv.org/abs/1502.03167)<br>
Reference : [#1](https://r2rt.com/implementing-batch-normalization-in-tensorflow.html)<br>
Reference : [#2](https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/38320613#38320613)<br>
Reference : [#3](https://stackoverflow.com/questions/40879967/how-to-use-batch-normalization-correctly-in-tensorflow)<br>

# Tensorflow Batch Normalization Method 

#1.[tf.nn.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization)<br>
#2.[tf.nn.fused_batch_norm](https://www.tensorflow.org/api_docs/python/tf/nn/fused_batch_norm)<br>
#3.[tf.layers.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)<br>
#4.[tf.contrib.layers.batch_norm](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm) (will deprecate)<br> 
#5.[tf.nn.batch_norm_with_global_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/batch_norm_with_global_normalization) (will deprecate)<br>


# **Usage** : 1 
### for Dense Layer(Fully Connecteted Layer , Affine Layer)
<pre> 
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
def batch_norm_0(x,train_phase,scope_bn):
    bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
    is_training=True,
    trainable=True,
    reuse=False,
    scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
    is_training=False,
    trainable=False,
    reuse=True,
    scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z
</pre>

For Inference Level 
MUST ADD THIS CODE
<pre>
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
</pre>


# **Usage** : 2 
### for Dense Layer(Fully Connecteted Layer , Affine Layer)
<pre> 
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
def batch_norm_1( _input , is_training , scope_bn):
    output = batch_norm(_input, scale=True, center=True , is_training=is_training , scope=scope_bn) # 테스트 시에도 학습이 되는건가
    return output
</pre>

For Inference Level 
MUST ADD THIS CODE
<pre>
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
</pre>


# **Usage** : 3_0 <br> 
### for Convolution Layer 
<pre> 
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
def batch_norm_2(x, phase_train , scope_bn):
    with tf.variable_scope(scope_bn):
        n_out=int(x.get_shape()[-1])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
</pre>
    
# **Usage** : 3_1
### for Dense Layer(Fully Connecteted Layer , Affine Layer)
<pre> 
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
def batch_norm_2(x, phase_train , scope_bn):
    with tf.variable_scope(scope_bn):
        n_out=int(x.get_shape()[-1])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


# Performace 
