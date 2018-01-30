import tensorflow as tf
a = tf.constant(5, tf.float32)
b = tf.constant(5, tf.float32)
with tf.Session() as sess:
   sess.run(tf.add(a, b)) # output is 10
