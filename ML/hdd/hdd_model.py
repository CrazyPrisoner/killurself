import numpy
import pandas
import tensorflow as tf


# Hyperparameters
learning_rate = 1e-4
batch_size = 100
training_iteration = 100000
num_inputs = 6
num_classes = 1
hidden1 = 128
hidden2 = 256
hidden3 = 512
hidden4 = 1024
# model parameters
sess = tf.InteractiveSession()
X = tf.placeholder(dtype=tf.float32,shape=[None, num_inputs],name="first_placeholder")
Y = tf.placeholder(dtype=tf.float32,shape=[None, num_classes], name="second_placeholder")

weight = tf.add()




}
