import tensorflow as tf

def convolution2d(name,x,out_ch,k=3 , s=2 , padding='SAME'):
    with tf.variable_scope(name) as scope:
        in_ch=x.get_shape()[-1]
        filter=tf.get_variable("w" , [k,k,in_ch , out_ch] , initializer=tf.contrib.layers.xavier_initializer())
        bias=tf.Variable(tf.constant(0.1) , out_ch)
        layer=tf.nn.conv2d(x , filter ,[1,s,s,1] , padding)+bias
        layer=tf.nn.relu(layer , name='relu')
        if __debug__ == True:
            print 'layer shape : ' ,layer.get_shape()

        return layer

def max_pool(x , k=3 , s=2 , padding='SAME'):

    if __debug__ ==True:
        print 'layer name :'
        print 'layer shape :',layer.get_shape()
    return tf.nn.max_pool(x , ksize=[1,k,k,1] , strides=[1,s,s,1] , padding=padding)


def affine(name,x,out_ch ,keep_prob):
    with tf.variable_scope(name) as scope:
        if len(x.get_shape())==4:
            batch, height , width , in_ch=x.get_shape().as_list()
            w_fc=tf.get_variable('w' , [height*width*in_ch ,out_ch] , initializer= tf.contrib.layers.xavier_initializer())
            x = tf.contrib.layers.flatten(x)
        elif len(x.get_shape())==2:
            batch, in_ch = x.get_shape().as_list()
            w_fc=tf.get_variable('w' ,[in_ch ,out_ch] ,initializer=tf.contrib.layers.xavier_initializer())

        b_fc=tf.Variable(tf.constant(0.1 ), out_ch)
        layer=tf.matmul(x , w_fc) + b_fc
        layer=tf.nn.relu(layer)
        layer=tf.nn.dropout(layer , keep_prob)
        print 'layer name :'
        print 'layer shape :',layer.get_shape()
        print 'layer dropout rate :',keep_prob
        return layer
def gap(name,x , n_classes ):
    in_ch=x.get_shape()[-1]
    gap_x=tf.reduce_mean(x, (1,2))
    with tf.variable_scope(name) as scope:
        gap_w=tf.get_variable('w' , shape=[in_ch , n_classes] , initializer=tf.random_normal_initializer(0,0.01) , trainable=True)
    y_conv=tf.matmul(gap_x, gap_w , name='y_conv')
    return y_conv

def algorithm(y_conv , y_ , learning_rate):
    """

    :param y_conv: logits
    :param y_: labels
    :param learning_rate: learning rate
    :return:  pred,pred_cls , cost , correct_pred ,accuracy
    """
    if __debug__ ==True:
        print y_conv.get_shape()
        print y_.get_shape()

    pred=tf.nn.softmax(y_conv , name='softmax')
    pred_cls=tf.argmax(pred , axis=1 , name='pred_cls')
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv , labels=y_) , name='cost')
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    correct_pred=tf.equal(tf.argmax(y_conv , 1) , tf.argmax(y_ , 1) , name='correct_pred')
    accuracy =  tf.reduce_mean(tf.cast(correct_pred , dtype=tf.float32) , name='accuracy')
    return pred,pred_cls , cost , train_op,correct_pred ,accuracy

if __name__ == '__main__':

    image_height, image_width, image_color_ch, n_classes,train_imgs, train_labs, test_imgs, test_labs=data.eye_64x64()

    x_ = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, image_color_ch] , name='x_')
    y_ = tf.placeholder(dtype=tf.int32, shape=[None, n_classes] ,name='y_')
    layer = convolution2d('conv1',x_,64,k=5)
    layer = max_pool(layer)
    layer = convolution2d('conv2',layer,64 )
    layer = max_pool(layer)
    layer = convolution2d('conv3',layer, 128)
    layer = max_pool(layer)
    top_conv = convolution2d('top_conv', x_, 128)
    layer = max_pool(top_conv)
    y_conv   = gap('gap' ,layer,n_classes)
    cam=get_class_map('gap',top_conv,0,im_width=image_width)
    pred, pred_cls, cost,train_op, correct_pred, accuracy=algorithm(y_conv , y_ ,0.005)
    saver=tf.train .Saver()
    sess=tf.Session()
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    try:
        saver.restore(sess, './model/best_acc.ckpt')
        print 'model was restored!'
    except tf.errors.NotFoundError:
        print 'there was no model'

    max_val=0
    check_point=1000
    for step in range(100):
        if step % check_point==0:
            inspect_cam(sess, cam , top_conv , test_imgs , test_labs,step , 50 , x_,y_,y_conv)
            val_acc, val_loss = sess.run([accuracy, cost], feed_dict={x_: test_imgs[:100], y_: test_labs[:100]})
            print np.shape(cam)
            print val_acc , val_loss
            if val_acc > max_val:
                saver.save(sess, './model/best_acc.ckpt')
                print 'model was saved!'
        batch_xs , batch_ys=batch.next_batch(train_imgs , train_labs , batch_size=60)
        train_acc, train_loss,_ =sess.run([accuracy,cost,train_op] , feed_dict={x_:batch_xs , y_:batch_ys})
