import os
import input_data
import tensorflow as tf
import model


data = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.variable_scope("regress"):
    x = tf.placeholder(tf.float32,[None, 784])
    y, varables = model.regreesion(x)

#train

y_ = tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

save = tf.trainSaver(variables)
