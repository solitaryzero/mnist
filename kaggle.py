import tensorflow as tf
import numpy as np
import random
import data
sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

import data
mnist = data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
answer = tf.reduce_sum(tf.argmax(y_conv,1))
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

input_file = open("train.csv",'r')
data_list = input_file.readlines()
pic_num = 0
pic = []
for line in data_list:
    tmp = line.split(',')
    if tmp[0] != "label":
        pic.append(tmp)
        for index in range(len(tmp)):
            pic[pic_num][index] = float(pic[pic_num][index])
            if (index != 0):
                pic[pic_num][index] = pic[pic_num][index]/255
        pic_num = pic_num+1
input_file.close()

pic_for_testing = pic[35000:]
pic = pic[0:35000]

for i in range(20000):
    chosen = random.sample(pic,100)
    chosen_pic = []
    chosen_label = []
    for index in range(len(chosen)):
        chosen_pic.append(chosen[index][1:])
        tmp_label = []
        for j in range(10):
            if (j == int(chosen[index][0])):
                tmp_label.append(1.0)
            else:
                tmp_label.append(0.0)
        chosen_label.append(tmp_label)

    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
        x:chosen_pic, y_: chosen_label, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: chosen_pic, y_: chosen_label, keep_prob: 0.5})
    if i%1000 == 0:
        save_path = saver.save(sess, "save_path/save_net.ckpt")

chosen_pic = []
chosen_label = []

index = 0
for k in range(min(1000,len(pic_for_testing)-index)):
    chosen_pic.append(pic_for_testing[index][1:])
    tmp_label = []
    for j in range(10):
        if (j == int(pic_for_testing[index][0])):
            tmp_label.append(1.0)
        else:
            tmp_label.append(0.0)
    chosen_label.append(tmp_label)
    index += 1

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: chosen_pic, y_:chosen_label, keep_prob: 1.0}))

input_file.close()


