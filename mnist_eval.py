import tensorflow as tf
import data
import numpy as np
#def ensemble(*paths):
#    for path in paths:




def eval_models(test_imgs , test_labs ,*model_paths):
    sess=tf.Session()
    softmax_=[]
    accuracy_=[]
    for model_path in model_paths:
        saver = tf.train.import_meta_graph(model_path + 'best_acc.ckpt.meta')
        saver.restore(sess,model_path+'best_acc.ckpt')
        tf.get_default_graph()
        softmax=tf.get_default_graph().get_tensor_by_name('softmax:0')
        accuracy=tf.get_default_graph().get_tensor_by_name('accuracy:0')
        #top_conv=tf.get_default_graph().get_tensor_by_name('top_conv/relu:0')
        x_=tf.get_default_graph().get_tensor_by_name('x_:0')
        y_=tf.get_default_graph().get_tensor_by_name('y_:0')
        phase_train=tf.get_default_graph().get_tensor_by_name('phase_train:0')
        softmax, accuracy = sess.run([softmax, accuracy], feed_dict={x_: test_imgs, y_: test_labs, phase_train: False})
        softmax_.append(softmax)
        accuracy_.append(accuracy)
    return softmax_ , accuracy_

if __name__=='__main__':
    model_path='./cnn_model/batch_norm/'
    model_path_1 = './cnn_model/non_batch_norm/'
    image_height, image_width, image_color_ch, n_classes, train_imgs, train_labs, test_imgs, test_labs = data.mnist_28x28()
    softmax_ , accuracy_=eval_models(test_imgs , test_labs ,model_path , model_path_1)
    print accuracy_
    """
    softmax,accuracy=sess.run([softmax,accuracy], feed_dict={x_:test_imgs , y_:test_labs , phase_train : False})
    print accuracy
    np_softmax=np.save('./softmax/batch_softmax',softmax)
    sess.close()
    batch_softmax_np=np.load('./softmax/batch_softmax.npy')
    non_batch_softmax_np = np.load('./softmax/non_batch_softmax.npy')
    ensemble_np=(np.reshape(batch_softmax_np,[10000,10]) +np.reshape(non_batch_softmax_np,[10000,10]))/2.
    print np.shape(batch_softmax_np)
    print np.shape(non_batch_softmax_np)
    #print ensemble_np
    err_np=np.ones([len(test_labs)])
    pred_cls=np.argmax(ensemble_np  ,axis=1)
    cls=np.argmax(test_labs , axis=1)
    err_np[pred_cls==cls]=0
    print np.mean(err_np)

    #model_path = './cnn_model/non_batch_norm/'
    #sess, x_, y_, phase_train, accuracy, softmax = restore_model(model_path)
    #image_height, image_width, image_color_ch, n_classes, train_imgs, train_labs, test_imgs, test_labs = data.mnist_28x28()
    #pred = sess.run([accuracy], feed_dict={x_: test_imgs, y_: test_labs, phase_train: False})
    """
