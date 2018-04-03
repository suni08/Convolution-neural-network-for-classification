
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 01:28:32 2017

@author: Sunita
"""
import numpy as np
import scipy
import tensorflow  as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing

#Extracting file from dataset
import gzip, zipfile
import sys, os
from PIL import Image

filename="img_align_celeba.zip"
label_filename="list_attr_celeba.txt"

#Defining height,width for resizing the images to 178x178
height=178
width=178

#Defining path for extracting dataset zip file
extract_path = "celeb_image_data"

#Defining image,label list
images = []
img_list = []
labels = []

#Extracting given dataset file    
with zipfile.ZipFile(filename, 'r') as zip:
    zip.extractall(extract_path)

#Extracting labels,images array needed for training    
for root, dirs, files in os.walk("."):
    path = root.split(os.sep)
        
    if "Numerals" in path:
        image_files = [fname for fname in files if fname.find(".png") >= 0]
        for file in image_files:
            #labels.append(int(path[-1]))
            images.append(os.path.join(*path, file))

#Resizing images    
for idx, imgs in enumerate(images):
    img = Image.open(imgs).convert('L') 
    img = img.resize((height, width), Image.ANTIALIAS)
    img_data = list(img.getdata())
    img_list.append(img_data)

#Storing image and labels in arrays to be used for training   
img_array = np.array(img_list)
img_array = np.subtract(255, USPS_img_array)
#USPS_label_array = np.array(labels)

#printing
#print(USPS_img_array)
#print(USPS_label_array)

#initialising hyperparams
batch_size=50
iterations=10

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#CNN using tensor flow
def tf_conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def tf_max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def tf_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def tf_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)

#first convolutional layer
x_p = tf.placeholder(tf.float32, shape=[None, 784])
W_conv1 = tf_weight_variable([5, 5, 1, 32])
b_conv1 = tf_bias_variable([32])
x_image = tf.reshape(x_p, [-1,28,28,1])
h_conv1 = tf.nn.relu(tf_conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = tf_max_pool_2x2(h_conv1)
    
#second convolutional layer
W_conv2 = tf_weight_variable([5, 5, 32, 64])
b_conv2 = tf_bias_variable([64])
h_conv2 = tf.nn.relu(tf_conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = tf_max_pool_2x2(h_conv2)
    
#Densely Connected Layer
W_fc1 = tf_weight_variable([7 * 7 * 64, 1024])
b_fc1 = tf_bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
#Readout
W_fc2 = tf_weight_variable([1024, 10])
b_fc2 = tf_bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
#Training
y_p = tf.placeholder(tf.float32, shape=[None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_p, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_p, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for idx in range(iterations):
        batch = np.split(img_array,batch_size)
        if idx % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x_p: batch[0],y_p: batch[1],keep_prob: 1.0})            
        train_step.run(feed_dict={x_p: batch[0], y_p: batch[1],keep_prob: 0.5})
    print('test accuracy %g' % accuracy.eval(feed_dict={x_p: img_array,y_p: labels_array,keep_prob: 1.0}))
    
plt.show()


