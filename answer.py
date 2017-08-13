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
load_path = saver.restore(sess, "save_path/save_net.ckpt")

test_file = open("test.csv",'r')
test_output_file = open("submission.csv",'w')
test_output_file.write("ImageId,Label\n")
testlines = test_file.readlines()
test = []
test_num = 0
for line in testlines:
    tmp = line.split(',')
    if tmp[0] != "pixel0":
        test.append(tmp)
        for index in range(len(tmp)):
            test[test_num][index] = float(test[test_num][index])/255
        test_num = test_num+1


for index in range(len(test)):
    chosen_pic = []
    chosen_label = []
    chosen_pic.append(test[index])
    tmp_label = []
    for j in range(10):
        tmp_label.append(0.0)
    chosen_label.append(tmp_label)
    index += 1

    test_output_file.write(str(index)+','+str(answer.eval(feed_dict={
        x: chosen_pic, y_:chosen_label, keep_prob: 1.0}))+'\n')

test_file.close()
test_output_file.close()


