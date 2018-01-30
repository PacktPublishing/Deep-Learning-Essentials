import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def cnn_model_fn(features, labels, mode):
    # Input Layer
    INPUT = tf.reshape(features["x"], [-1, 28, 28, 1])
    # Conv-1 Layer
    CONV1 = tf.layers.conv2d(
        inputs=INPUT,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Pool-1 Layer
    POOL1 = tf.layers.max_pooling2d(inputs=CONV1, pool_size=[2, 2], strides=2)
    # Conv-2 Layer
    CONV2 = tf.layers.conv2d(
        inputs=POOL1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Pool-2 Layer
    POOL2 = tf.layers.max_pooling2d(inputs=CONV2, pool_size=[2, 2], strides=2)
    # Pool-2 Flattened Layer
    POOL2_FLATTENED = tf.reshape(POOL2, [-1, 7 * 7 * 64])
    FC1 = tf.layers.dense(inputs=POOL2_FLATTENED, units=1024, activation=tf.nn.relu)
    # Dropout Layer
    DROPOUT = tf.layers.dropout(
        inputs=FC1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    FC2 = tf.layers.dense(inputs=DROPOUT, units=10)
    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=FC2)
    # Configure the Training Op (for TRAIN mode)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


if __name__ == "__main__":
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    features = mnist.train.images
    labels = np.asarray(mnist.train.labels, dtype=np.int32)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": features},
      y=labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
    mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
    mnist_classifier.train(input_fn=train_input_fn,steps=20000)
