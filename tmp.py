import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
img_0=Image.open('./color_img.png')
img_1=Image.open('./color_img.png')
img_0=np.expand_dims(img_0 , 0)
img_1 = np.expand_dims(img_1 , 0)
imgs=np.vstack([img_0 , img_1 ])
print np.shape(imgs)
img=np.asarray(imgs)
tensor_img= tf.Variable(img)
tensor_img=tf.reduce_mean(tensor_img , axis = [3])
sess=tf.Session()
sess.run(tf.global_variables_initializer())
img=sess.run(tensor_img)
print img



