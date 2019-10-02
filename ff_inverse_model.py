"""Inverse model implemented using a feedforward network.
    input: perception (vision) at time t-1 and at time t
    output: action at time t-1
"""

import numpy as np
import tensorflow as tf
import scipy.io
import math
import sys

import matplotlib.pyplot as plt

logs_path = './logs/ff_forward_model'


print("Loading dataset...")

a = scipy.io.loadmat("matlab/database/final_database_train.mat")
X_init = 1*a["final_database_train"]
X_augm_train = X_init[:1845,:] #np.append(X_train_all,X_train_no_mc,axis=0)
print(X_augm_train.shape)
n_samples = X_augm_train.shape[0]
##########################################################################################


def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(1.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(1.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),  minval=low, maxval=high,  dtype=tf.float64)
    #stddev = np.sqrt(1.0 / (fan_in + fan_out))
    #return tf.random_normal((fan_in, fan_out), mean = 0.0, stddev=stddev, dtype=tf.float64)



def shuffle_data(x):
    y = list(x)
    np.random.shuffle(y)
    return np.asarray(y)


    

def main():
    # param_id= float(sys.argv[1:][0])
    
    learning_rate = 0.01 #param_id # 
    batch_size = 1000
    training_epochs = 2000
    display_step = 1

    # Write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    n_input_features = 20
    n_output_features = 4
    n_hidden = 150

    tx = tf.placeholder(tf.float32, (None, n_input_features))
    ty = tf.placeholder(tf.float32, (None, n_output_features))
    
    tW1 = tf.get_variable("W1_im", shape=(n_input_features, n_hidden))
    tW2 = tf.get_variable("W2_im", shape=(n_hidden, n_hidden))
    tW3 = tf.get_variable("W3_im", shape=(n_hidden, n_output_features))
    
    tO1 = tf.nn.tanh(tf.matmul(tx, tW1))
    tO2 = tf.nn.tanh(tf.matmul(tO1, tW2))
    tO3 = tf.matmul(tO2, tW3)
    
    tMSE = tf.reduce_mean(tf.square(tf.subtract(tO3, ty)))
    
    tOptimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    tOptimize = tOptimizer.minimize(tMSE)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size) 

            X_shuffled = shuffle_data(X_augm_train)

            # Loop over all batches
            for i in range(total_batch):


                batch_xs_augmented = X_shuffled[batch_size*i:batch_size*i+batch_size] 
                
                batch_xs   = np.asarray([item[:28]   for item in batch_xs_augmented])#np.asarray([item[:18]   for item in batch_xs_augmented])
                batch_xs_noiseless   = np.asarray([item[28:]   for item in batch_xs_augmented])#np.asarray([item[:18]   for item in batch_xs_augmented])
                            # batch_xs_noiseless_J   = np.asarray([item[8:12]   for item in batch_xs_noiseless])

                x = batch_xs[:,:20] #batch_xs_noiseless[:,:24] #[:,8:16]
                y = batch_xs_noiseless[:,24:]
                
                # Fit training using batch data
                _, cost, fm_output = sess.run((tOptimize, tMSE, tO2), 
                    feed_dict={tx: x, ty: y})


                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch: %04d / %04d, Cost= %04f" % (epoch,training_epochs,avg_cost))

        save_path = saver.save(sess, "./models/ff_im_compl_data.ckpt")

    
if __name__ == "__main__":
    main()
