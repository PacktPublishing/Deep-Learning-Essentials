import gym
import numpy as np
import random
import tensorflow as tf


if __name__ == "__main__":

    # Set parameters
    y = .99
    e = 0.2
    num_episodes = 5000
    num_steps = 100
    num_states = 16
    sample_epsilon = 0.2
    eta = 0.05


    # Setup environment
    env = gym.make('FrozenLake-v0')
    tf.reset_default_graph()

    # Define the tensor graph
    input_matrix = tf.placeholder(shape=[1,16],dtype=tf.float32)
    weight_matrix = tf.Variable(tf.random_uniform([16,4],0,0.01))
    Q_matrix = tf.matmul(input_matrix,weight_matrix)
    prediction_matrix = tf.argmax(Q_matrix,1)
    nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Q_matrix))
    train = tf.train.GradientDescentOptimizer(learning_rate=eta)
    model = train.minimize(loss)
    init_op = tf.global_variables_initializer()


    #create lists to contain total rewards and steps per episode
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(num_episodes):
            #Reset environment and get first new observation
            current_state = env.reset()
            done = False
            current_step = 0
            #The Q-Network
            while current_step < num_steps:
                current_step = current_step + 1
                #Choose an action by greedily (with e chance of random action) from the Q-network
                ip_q = np.zeros(num_states)
                ip_q[current_state] = 1
                a,allQ = sess.run([prediction_matrix,Q_matrix],feed_dict={input_matrix:[ip_q]})
                if np.random.rand(1) < sample_epsilon:
                    a[0] = env.action_space.sample()
                #Get new state and reward from environment
                next_state, reward, done, info = env.step(a[0])
                #Obtain the Q' values by feeding the new state through our network
                ip_q1 = np.zeros(num_states)
                ip_q1[next_state] = 1
                Q1 = sess.run(Q_matrix,feed_dict={input_matrix:[ip_q1]})
                #Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a[0]] = reward + y*maxQ1
                #Train our network using target and predicted Q values
                _,W1 = sess.run([model,weight_matrix],feed_dict={input_matrix:[ip_q],nextQ:targetQ})
                if current_step % 50 == 0:
                    print("Next State = {}, Reward = {}, Q-Value = {}".format(next_state, reward, maxQ1))
                current_state = next_state
                if done == True:
                    break
