#loadmodel.py

import numpy as np
import tensorflow as tf
import scipy.io
import math

import matplotlib.pyplot as plt

#*****************************#
from train_all import VariationalAutoencoder
from train_all import network_param
#*****************************#

print("Loading dataset...")

# a = scipy.io.loadmat("matlab/database/n_database.mat")
a = scipy.io.loadmat("matlab/database/final_database_all0_test.mat")#
X_init = 1*a["final_database_test"]


X_augm_test = X_init #np.append(X_train_all,X_train_no_mc,axis=0)
print(X_augm_test.shape)



################################################################################################################
with tf.Graph().as_default() as g:
	with tf.Session() as sess:

		# Network parameters
		network_architecture = network_param()
		learning_rate = 0.00001
		batch_size = 1000


		model = VariationalAutoencoder(sess,network_architecture, batch_size=batch_size, learning_rate=learning_rate, vae_mode=False, vae_mode_modalities=False)

	with tf.Session() as sess:
		new_saver = tf.train.Saver() 

		new_saver.restore(sess, "./models/simple_all.ckpt")
		print("Model restored.")
						


		# Test 1: complete data
		print('Test 1')
		sample_init = 100
		x_sample = X_augm_test[sample_init:sample_init+batch_size,:34]  #X_3[sample_init:sample_init+batch_size,:2] #
		x_reconstruct_M, x_reconstruct_log_sigma_sq_M = model.reconstruct_M(sess,x_sample)
		# x_reconstruct_J, x_reconstruct_log_sigma_sq_J = model.reconstruct_J(sess,x_sample)

		scipy.io.savemat("results/mvae_all_test1.mat",{"x_reconstruct":x_reconstruct_M,"x_reconstruct_log_sigma_sq":x_reconstruct_log_sigma_sq_M,"x_sample":x_sample})
		# scipy.io.savemat("results/mvae_test1_J.mat",{"x_reconstruct":x_reconstruct_J,"x_reconstruct_log_sigma_sq":x_reconstruct_log_sigma_sq_J,"x_sample":x_sample})



		################################################################################################################

                
		# Test 2: 
		print('Test 2')
		x_sample_nv_1 = X_augm_test[sample_init:sample_init+batch_size,:12] 
		x_sample_nv_2 = X_augm_test[sample_init:sample_init+batch_size,12:24] 
		x_sample_nv_3 = X_augm_test[sample_init:sample_init+batch_size,24:27] 
		x_sample_nv_4 = X_augm_test[sample_init:sample_init+batch_size,27:30] 
		x_sample_nv_5 = np.full((x_sample_nv_1.shape[0],4),0)  
		x_sample_nv = np.append(x_sample_nv_1,np.append(x_sample_nv_2,np.append(x_sample_nv_3,np.append(x_sample_nv_4,x_sample_nv_5,axis=1),axis=1),axis=1),axis=1)
                

		x_reconstruct, x_reconstruct_log_sigma_sq = model.reconstruct_M(sess,x_sample_nv) 
		
		scipy.io.savemat("results/mvae_all_test2.mat",{"x_reconstruct":x_reconstruct,"x_reconstruct_log_sigma_sq":x_reconstruct_log_sigma_sq,"x_sample":x_sample_nv})


        

                
		# Test 3: 
		print('Test 3')
		x_sample_nv_1 = X_augm_test[sample_init:sample_init+batch_size,:12] 
		x_sample_nv_2 = X_augm_test[sample_init:sample_init+batch_size,12:24] 
		x_sample_nv_3 = np.full((x_sample_nv_1.shape[0],3),0)  
		x_sample_nv_4 = np.full((x_sample_nv_1.shape[0],3),0)  
		x_sample_nv_5 = np.full((x_sample_nv_1.shape[0],4),0)  
		x_sample_nv = np.append(x_sample_nv_1,np.append(x_sample_nv_2,np.append(x_sample_nv_3,np.append(x_sample_nv_4,x_sample_nv_5,axis=1),axis=1),axis=1),axis=1)
                

		x_reconstruct, x_reconstruct_log_sigma_sq = model.reconstruct_M(sess,x_sample_nv) 
		
		scipy.io.savemat("results/mvae_all_test3.mat",{"x_reconstruct":x_reconstruct,"x_reconstruct_log_sigma_sq":x_reconstruct_log_sigma_sq,"x_sample":x_sample_nv})


        
		# Test 4: 
		print('Test 4')
		x_sample_nv_1 = np.full((x_sample_nv_1.shape[0],12),-2)  
		x_sample_nv_2 = X_augm_test[sample_init:sample_init+batch_size,12:24] 
		x_sample_nv_3 = np.full((x_sample_nv_1.shape[0],3),0)  
		x_sample_nv_4 = np.full((x_sample_nv_1.shape[0],3),0)  
		x_sample_nv_5 = np.full((x_sample_nv_1.shape[0],4),0)  
		x_sample_nv = np.append(x_sample_nv_1,np.append(x_sample_nv_2,np.append(x_sample_nv_3,np.append(x_sample_nv_4,x_sample_nv_5,axis=1),axis=1),axis=1),axis=1)
                

		x_reconstruct, x_reconstruct_log_sigma_sq = model.reconstruct_M(sess,x_sample_nv) 
		
		scipy.io.savemat("results/mvae_all_test4.mat",{"x_reconstruct":x_reconstruct,"x_reconstruct_log_sigma_sq":x_reconstruct_log_sigma_sq,"x_sample":x_sample_nv})


        

        
		# Test 5: 
		print('Test 5')
		x_sample_nv_1 = X_augm_test[sample_init:sample_init+batch_size,:12] 
		x_sample_nv_2 = np.full((x_sample_nv_1.shape[0],12),0) 
		x_sample_nv_3 = np.full((x_sample_nv_1.shape[0],3),0) 
		x_sample_nv_4 = X_augm_test[sample_init:sample_init+batch_size,27:30] 
		x_sample_nv_5 = np.full((x_sample_nv_1.shape[0],4),0)  
		x_sample_nv = np.append(x_sample_nv_1,np.append(x_sample_nv_2,np.append(x_sample_nv_3,np.append(x_sample_nv_4,x_sample_nv_5,axis=1),axis=1),axis=1),axis=1)
                

		x_reconstruct, x_reconstruct_log_sigma_sq = model.reconstruct_M(sess,x_sample_nv) 
		
		scipy.io.savemat("results/mvae_all_test5.mat",{"x_reconstruct":x_reconstruct,"x_reconstruct_log_sigma_sq":x_reconstruct_log_sigma_sq,"x_sample":x_sample_nv})




        
		# Test 6: 
		print('Test 6')
		x_sample_nv_1 = np.full((x_sample_nv_1.shape[0],12),0) 
		x_sample_nv_2 = X_augm_test[sample_init:sample_init+batch_size,12:24] 
		x_sample_nv_3 = np.full((x_sample_nv_1.shape[0],3),0)
		x_sample_nv_4 = X_augm_test[sample_init:sample_init+batch_size,27:30] 
		x_sample_nv_5 = np.full((x_sample_nv_1.shape[0],4),0)  
		x_sample_nv = np.append(x_sample_nv_1,np.append(x_sample_nv_2,np.append(x_sample_nv_3,np.append(x_sample_nv_4,x_sample_nv_5,axis=1),axis=1),axis=1),axis=1)
                

		x_reconstruct, x_reconstruct_log_sigma_sq = model.reconstruct_M(sess,x_sample_nv) 
		
		scipy.io.savemat("results/mvae_all_test6.mat",{"x_reconstruct":x_reconstruct,"x_reconstruct_log_sigma_sq":x_reconstruct_log_sigma_sq,"x_sample":x_sample_nv})



        
		# Test 7: 
		print('Test 7')
		x_sample_nv_1 = np.full((x_sample_nv_1.shape[0],12),0) 
		x_sample_nv_2 = np.full((x_sample_nv_1.shape[0],12),0) 
		x_sample_nv_3 = np.full((x_sample_nv_1.shape[0],3),0)
		x_sample_nv_4 = X_augm_test[sample_init:sample_init+batch_size,27:30] 
		x_sample_nv_5 = np.full((x_sample_nv_1.shape[0],4),0)  
		x_sample_nv = np.append(x_sample_nv_1,np.append(x_sample_nv_2,np.append(x_sample_nv_3,np.append(x_sample_nv_4,x_sample_nv_5,axis=1),axis=1),axis=1),axis=1)
                

		x_reconstruct, x_reconstruct_log_sigma_sq = model.reconstruct_M(sess,x_sample_nv) 
		
		scipy.io.savemat("results/mvae_all_test7.mat",{"x_reconstruct":x_reconstruct,"x_reconstruct_log_sigma_sq":x_reconstruct_log_sigma_sq,"x_sample":x_sample_nv})


