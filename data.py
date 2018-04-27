from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np

def mnist_28x28():
    mnist = input_data.read_data_sets('MNIST_DATA_SET', one_hot=True)
    mnist_train_imgs=np.reshape(mnist.train.images , (55000 ,28,28,1))
    mnist_train_labs=mnist.train.labels
    mnist_test_imgs = np.reshape(mnist.test.images, (10000, 28, 28, 1))
    mnist_test_labs = mnist.test.labels

    print mnist_test_imgs.shape , mnist_train_imgs.shape , mnist_train_labs.shape , mnist_test_labs.shape

    image_height = 28
    image_width = 28
    image_color_ch = 1
    n_classes = 10
    train_imgs=mnist_train_imgs
    train_labs=mnist_train_labs
    test_imgs=mnist_test_imgs
    test_labs=mnist_test_labs
    return image_height , image_width , image_color_ch , n_classes, train_imgs , train_labs , test_imgs, test_labs

def next_batch(imgs, labs , batch_size):
    indices=random.sample(range(np.shape(imgs)[0]) , batch_size)
    batch_xs=imgs[indices]
    batch_ys=labs[indices]
    return batch_xs , batch_ys