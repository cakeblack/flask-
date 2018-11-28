import tensorflow as tf

#Y = W * x + b
def regression(x):
    W = tf.Variable(tf.zeros([784, 10])),
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.sotfmax(tf.matumul(x, W) + b)
    return y, [W, b]