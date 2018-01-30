import numpy as np
import tensorflow as tf


def lstm_model(word_vectors, embedding_dimension = 64):

    # Define configurations
    batch_size = 4
    lstm_units = 16
    num_classes = 2
    max_sequence_length = 4
    num_iterations = 1000

    # Define tensor graph
    labels = tf.placeholder(tf.float64, [batch_size, num_classes])
    raw_data = tf.placeholder(tf.int32, [batch_size, max_sequence_length])
    data = tf.Variable(tf.zeros([batch_size, max_sequence_length, embedding_dimension]),dtype=tf.float32)
    data = tf.nn.embedding_lookup(word_vectors,raw_data)
    weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    wrapped_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.8)
    output, state = tf.nn.dynamic_rnn(wrapped_lstm_cell, data, dtype=tf.float64)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    weight = tf.cast(weight, tf.float64)
    last = tf.cast(last, tf.float64)
    bias = tf.cast(bias, tf.float64)
    prediction = (tf.matmul(last, weight) + bias)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Generate toy data set
    # Assume word_vectors is a trained word embedding model with 1024 words
    # toy_data is a list of sequences and toy label is a list of output labels
    # toy_data:[[0,1,2,3]] : 1 seq len = 4, each value is index in word_vectors
    # toy_label:[[1,0]] -> probability of class '1' is 1 and class '2' is 0

    toy_data = [[0,1,2,3],[10,11,12,13],[20,21,22,23],[30,31,32,33]]
    toy_label = [[1,0],[0,1],[1,0],[0,1]]

    # Run the graph with toy dataset

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(optimizer, {raw_data: toy_data, labels: toy_label})
    return optimizer

if __name__ == "__main__":

    # Define a random word_vectors model for this tutorial
    embedding_dimension = 64
    word_vectors = np.random.random([1024, embedding_dimension])
    lstm_model(word_vectors)

