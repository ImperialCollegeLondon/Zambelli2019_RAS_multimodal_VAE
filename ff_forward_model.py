"""Forward model implemented using a feedforward network.
    input: perception (all modalities) at time t-1 and action at time t-1
    output: perception (all modalities) at time t
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
    
    learning_rate = 0.00005
    batch_size = 1000
    training_epochs = 5000
    display_step = 1

    # Write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    n_input_features = 14
    n_output_features = 10

    tx = tf.placeholder(tf.float32, (None, n_input_features))
    ty = tf.placeholder(tf.float32, (None, n_output_features))
    
    tW1 = tf.get_variable("W1_fm", shape=(n_input_features, n_input_features))
    tW2 = tf.get_variable("W2_fm", shape=(n_input_features, n_output_features))
    
    tO1 = tf.nn.tanh(tf.matmul(tx, tW1))
    tO2 = tf.matmul(tO1, tW2)
    
    tMSE = tf.reduce_mean(tf.square(tf.subtract(tO2, ty)))
    
    tOptimizer = tf.train.AdamOptimizer()
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

                input_idx = [4,5,6,7, 12,13,14,15, 17, 19, 24,25,26,27]
                input_idx_t1 = [0,1,2,3,  8,9,10,11,  16, 18]

                x = batch_xs[:,input_idx] #batch_xs_noiseless[:,input_idx]
                y = batch_xs_noiseless[:,input_idx_t1]
                
                # Fit training using batch data
                _, cost, fm_output = sess.run((tOptimize, tMSE, tO2), 
                    feed_dict={tx: x, ty: y})


                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch: %04d / %04d, Cost= %04f" % (epoch,training_epochs,avg_cost))

        save_path = saver.save(sess, "./models/ff_fm_compl_data.ckpt")

    
if __name__ == "__main__":
    main()
