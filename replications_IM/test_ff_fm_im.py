"""
Test vae forward and inverse models:
 - test that generated outputs match dataset
 - test concatenation IM -> FM: vision -> IM -> action -> FM -> vision'
"""

import numpy as np
import tensorflow as tf
import scipy.io
import math
import sys



import matplotlib.pyplot as plt


print("Loading dataset...")

# a = scipy.io.loadmat("matlab/database/n_database.mat")
a = scipy.io.loadmat("matlab/database/final_database_test.mat")
X_init = 1*a["final_database_test"]
X_augm_test = X_init[:1845,:] #
print(X_augm_test.shape)



with tf.Graph().as_default() as g:


	batch_size = 100

	n_input_features = 14
	n_output_features = 10

	W1_fm = tf.get_variable("W1_fm", shape=(n_input_features, n_input_features))
	W2_fm = tf.get_variable("W2_fm", shape=(n_input_features, n_output_features))

	saver_fm = tf.train.Saver()


	# Test forward model

	with tf.Session() as sess:
	  # Restore variables from disk.
	  saver_fm.restore(sess, "./models/ff_fm_compl_data.ckpt")
	  print("Model restored.")

	  sample_init = 100
	  input_idx = [4,5,6,7, 12,13,14,15, 17, 19, 24,25,26,27]
	  x_test = X_augm_test[sample_init:sample_init+batch_size,input_idx]
	  y_test = X_augm_test[sample_init:sample_init+batch_size,input_idx[:10]]

	  W1_fm = tf.cast(W1_fm, dtype=tf.float64)
	  W2_fm = tf.cast(W2_fm, dtype=tf.float64)

	  y_ff_fm = tf.matmul(tf.nn.tanh(tf.matmul(x_test, W1_fm)),W2_fm)

	  MSE = tf.reduce_mean(tf.square(tf.subtract(y_ff_fm, y_test)))

	  mse_ff_fm, y_ff_fm_res, W1_fm_res, W2_fm_res = sess.run((MSE,y_ff_fm,W1_fm,W2_fm))
	  print('MSE ff FM = ',mse_ff_fm)

	  scipy.io.savemat("results/ff_fm_compl_data_test.mat",{"y_test":y_test,"y_ff_fm":y_ff_fm_res})

# print('W1_fm',W1_fm_res)
# print('W2_fm',W2_fm_res)

with tf.Graph().as_default() as g:


	# Test inverse model

	batch_size = 100

	n_input_features = 20
	n_output_features = 4
	n_hidden = 150

	W1_im = tf.get_variable("W1_im", shape=(n_input_features, n_hidden))
	W2_im = tf.get_variable("W2_im", shape=(n_hidden, n_hidden))
	W3_im = tf.get_variable("W3_im", shape=(n_hidden, n_output_features))

	saver_im = tf.train.Saver()

	with tf.Session() as sess:
	  # Restore variables from disk.
	  saver_im.restore(sess, "./models/ff_im_compl_data.ckpt")
	  print("Model restored.")

	  sample_init = 100
	  x_test = X_augm_test[sample_init:sample_init+batch_size,:20]
	  y_test = X_augm_test[sample_init:sample_init+batch_size,24:]

	  W1_im = tf.cast(W1_im, dtype=tf.float64)
	  W2_im = tf.cast(W2_im, dtype=tf.float64)
	  W3_im = tf.cast(W3_im, dtype=tf.float64)

	  tO1 = tf.nn.tanh(tf.matmul(x_test, W1_im))
	  tO2 = tf.nn.tanh(tf.matmul(tO1, W2_im))
	  y_ff_im = tf.matmul(tO2, W3_im)

	  MSE = tf.reduce_mean(tf.square(tf.subtract(y_ff_im, y_test)))

	  mse_ff_im, y_ff_im_res, W1_im_res, W2_im_res, W3_im_res = sess.run((MSE,y_ff_im,W1_im,W2_im,W3_im))
	  print('MSE ff IM = ',mse_ff_im)

	  scipy.io.savemat("results/ff_im_compl_data_test.mat",{"y_test":y_test,"y_ff_im":y_ff_im_res})


with tf.Graph().as_default() as g:


	# Test inverse model on control task, i.e. only vision input, compare u_tm1

	batch_size = 100

	n_input_features = 20
	n_output_features = 4
	n_hidden = 150

	W1_im = tf.get_variable("W1_im", shape=(n_input_features, n_hidden))
	W2_im = tf.get_variable("W2_im", shape=(n_hidden, n_hidden))
	W3_im = tf.get_variable("W3_im", shape=(n_hidden, n_output_features))

	saver_im = tf.train.Saver()

	with tf.Session() as sess:
	  # Restore variables from disk.
	  saver_im.restore(sess, "./models/ff_im_compl_data.ckpt")
	  print("Model restored.")

	  sample_init = 100
	  x_sample = X_augm_test[sample_init:sample_init+batch_size,:28]
	  x_sample_nv_1 = np.full((x_sample.shape[0],4),-2)
	  x_sample_nv_2 = X_augm_test[sample_init:sample_init+batch_size,4:16]
	  x_sample_nv_3 = np.full((x_sample.shape[0],4),-2)
	  x_sample_nv = np.append( x_sample_nv_1, np.append( x_sample_nv_2, x_sample_nv_3, axis=1), axis=1)
	  x_test = x_sample_nv
	  y_test = X_augm_test[sample_init:sample_init+batch_size,24:]

	  W1_im = tf.cast(W1_im, dtype=tf.float64)
	  W2_im = tf.cast(W2_im, dtype=tf.float64)
	  W3_im = tf.cast(W3_im, dtype=tf.float64)

	  tO1 = tf.nn.tanh(tf.matmul(x_test, W1_im))
	  tO2 = tf.nn.tanh(tf.matmul(tO1, W2_im))
	  y_ff_im = tf.matmul(tO2, W3_im)

	  MSE = tf.reduce_mean(tf.square(tf.subtract(y_ff_im, y_test)))

	  mse_ff_im, y_ff_im_res, W1_im_res, W2_im_res, W3_im_res = sess.run((MSE,y_ff_im,W1_im,W2_im,W3_im))
	  print('MSE ff IM CL test = ',mse_ff_im)

	  scipy.io.savemat("results/ff_im_cl_compl_data_test.mat",{"y_test":y_test,"y_ff_im":y_ff_im_res})



# Test concatenation

with tf.Session() as sess:
	
	sample_init = 100
	input_idx = [4,5,6,7, 12,13,14,15, 17, 19, 24,25,26,27]

	x_sample_qv = X_augm_test[sample_init:sample_init+batch_size,input_idx[:8]]
	x_sample_ps = np.full((x_sample.shape[0],2),-2)
	x_sample_tofm = np.append( x_sample_qv, np.append( x_sample_ps, y_ff_im_res, axis=1), axis=1)


	y_ff_imfm = tf.matmul(tf.nn.tanh(tf.matmul(x_sample_tofm, W1_fm_res)),W2_fm_res)


	y_test = X_augm_test[sample_init:sample_init+batch_size,input_idx[:10]]

	MSE = tf.reduce_mean(tf.square(tf.subtract(y_ff_imfm, y_test)))

	MSE_concat, y_ff_imfm_res = sess.run((MSE,y_ff_imfm))

	print('MSE_concat',MSE_concat)

	scipy.io.savemat("results/ff_imfm_compl_data_test.mat",{"y_test":y_test,"y_ff_imfm":y_ff_imfm_res})




	# vision_input = X_augm_test[sample_init:sample_init+batch_size,8:16]
	# vision_test = vision_input[:,:4] # ?????
	# im_input = X_augm_test[sample_init:sample_init+batch_size,:20]

	# tO1 = tf.nn.tanh(tf.matmul(im_input, W1_im_res))
	# tO2 = tf.nn.tanh(tf.matmul(tO1, W2_im_res))
	# action_im = tf.matmul(tO2, W3_im_res)
	# # add state to actions (i.e. q_{t-1}, v_{t-1}, p_{t-1}, s_{t-1})
	# action_augm = tf.concat([X_augm_test[sample_init:sample_init+batch_size,[4,5,6,7, 12,13,14,15, 17, 19]],action_im],1)
	
	# vision_fm = tf.matmul(tf.nn.tanh(tf.matmul(action_augm, W1_fm_res)),W2_fm_res)
    

	# MSE = tf.reduce_mean(tf.square(tf.subtract(vision_fm[:,:4], vision_test)))

	# MSE_concat, y_ff_imfm_res = sess.run((MSE,vision_fm))	

	# print('MSE_concat',MSE_concat)

	# scipy.io.savemat("results/ff_imfm_compl_data_test.mat",{"y_test":vision_test,"y_ff_imfm":y_ff_imfm_res})
