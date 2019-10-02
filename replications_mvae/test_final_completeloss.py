#loadmodel.py

import numpy as np
import tensorflow as tf
import scipy.io
import math
import sys

sys.path.append('../../')

import matplotlib.pyplot as plt

#*****************************#
from train_final_completeloss import VariationalAutoencoder
from train_final_completeloss import network_param
#*****************************#


print("Loading dataset...")

# a = scipy.io.loadmat("matlab/database/n_database.mat")
a = scipy.io.loadmat("matlab/database/final_database_test.mat")
X_init = 1*a["final_database_test"]
X_augm_test = X_init
print(X_augm_test.shape)



#print("Loading Maxime kinect data...")
## b = scipy.io.loadmat("data_dancing_iCub.mat")
## X_test = 1*b["n_X_dancing_iCub"] 
#b = scipy.io.loadmat("matlab/database/dataMaximeKinect.mat")
#X_test_Max = 1*b["maxime_data"] 
#b = scipy.io.loadmat("matlab/database/dataMaximeKinect_4_rescaled.mat")
#X_test_Max_4_rescaled = 1*b["m"] 
#b = scipy.io.loadmat("matlab/database/dataMaximeKinect_5_rescaled.mat")
#X_test_Max_5_rescaled = 1*b["m"] 
#b = scipy.io.loadmat("matlab/database/dataMaximeKinect_7_rescaled.mat")
#X_test_Max_7_rescaled = 1*b["m"] 
#b = scipy.io.loadmat("matlab/database/dataMaximeKinect_8_rescaled.mat")
#X_test_Max_8_rescaled = 1*b["m"] 
#
#print("Loading Martina Piano kinect data...")
## b = scipy.io.loadmat("data_dancing_iCub.mat")
## X_test = 1*b["n_X_dancing_iCub"] 
#b = scipy.io.loadmat("matlab/database/dataMartinaPianoKinect.mat")
#X_test_MartinaPiano = 1*b["martina_piano_hand"] 
#b = scipy.io.loadmat("matlab/database/dataMartinaPianoKinect_rescaled.mat")
#X_test_MartinaPiano_rescaled = 1*b["database"] 
#b = scipy.io.loadmat("matlab/database/dataMartinaPiano_aligned2D.mat")
#X_test_MartinaPiano_aligned2D = 1*b["dataMartinaPiano_aligned2D"] 




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
      param_id= sys.argv[1:][0]
      new_saver.restore(sess, "./models/final_completeloss_all_conf_"+param_id+".ckpt")
      print("Model restored.")
                        


      # Test 1: complete data
      print('Test 1')
      sample_init = 100
      x_sample = X_augm_test[sample_init:sample_init+batch_size,:28]  
      x_reconstruct, x_reconstruct_log_sigma_sq = model.reconstruct(sess,x_sample)
      scipy.io.savemat("results/mvae_final_completeloss_test1.mat",{"x_reconstruct":x_reconstruct,"x_reconstruct_log_sigma_sq":x_reconstruct_log_sigma_sq,"x_sample":x_sample})

      ################################################################################################################

                
      # Test 2: vision only
      print('Test 2')
      x_sample_nv_1 = np.full((x_sample.shape[0],8),-2)  
      x_sample_nv_2 = X_augm_test[sample_init:sample_init+batch_size,8:16] 
      x_sample_nv_3 = np.full((x_sample.shape[0],12),-2)  
      x_sample_nv = np.append( x_sample_nv_1, np.append( x_sample_nv_2, x_sample_nv_3, axis=1), axis=1)
                

      x_reconstruct, x_reconstruct_log_sigma_sq = model.reconstruct(sess,x_sample_nv) 
      scipy.io.savemat("results/mvae_final_completeloss_test2.mat",{"x_reconstruct":x_reconstruct,"x_reconstruct_log_sigma_sq":x_reconstruct_log_sigma_sq,"x_sample":x_sample_nv})


                
      # Test 3: joint (t) and vision -- for control loop
      print('Test 3')
      x_sample_nv_1 = np.full((x_sample.shape[0],4),-2)
      x_sample_nv_2 = X_augm_test[sample_init:sample_init+batch_size,4:16]
      x_sample_nv_3 = np.full((x_sample.shape[0],12),-2)
      x_sample_nv = np.append( x_sample_nv_1, np.append( x_sample_nv_2, x_sample_nv_3, axis=1), axis=1)
                                                                               
        
      x_reconstruct, x_reconstruct_log_sigma_sq = model.reconstruct(sess,x_sample_nv) 
      scipy.io.savemat("results/mvae_final_completeloss_test3.mat",{"x_reconstruct":x_reconstruct,"x_reconstruct_log_sigma_sq":x_reconstruct_log_sigma_sq,"x_sample":x_sample_nv})


        
      # Test 4: only data at time t -- for prediction ?
      print('Test 4')
      x_sample_nv_1 = np.full((x_sample.shape[0],4),-2)  
      x_sample_nv_2 = X_augm_test[sample_init:sample_init+batch_size,4:8] 
      x_sample_nv_3 = np.full((x_sample.shape[0],4),-2)  
      x_sample_nv_4 = X_augm_test[sample_init:sample_init+batch_size,12:16] 
      x_sample_nv_5 = np.full((x_sample.shape[0],1),-2)  
      x_sample_nv_6 = X_augm_test[sample_init:sample_init+batch_size,17:18] 
      x_sample_nv_7 = np.full((x_sample.shape[0],1),-2)  
      x_sample_nv_8 = X_augm_test[sample_init:sample_init+batch_size,19:20]
      x_sample_nv_9 = np.full((x_sample.shape[0],4),-2)
      x_sample_nv_10 = X_augm_test[sample_init:sample_init+batch_size,24:] 
      x_sample_nv = np.append(x_sample_nv_1,
                                        np.append(x_sample_nv_2,
                                                  np.append(x_sample_nv_3,
                                                            np.append(x_sample_nv_4,
                                                                      np.append(x_sample_nv_5,
                                                                                np.append(x_sample_nv_6,
                                                                                          np.append(x_sample_nv_7,
                                                                                                    np.append(x_sample_nv_8,
                                                                                                              np.append(x_sample_nv_9,
                                                                                                                        x_sample_nv_10,axis=1)
                                                                                                              ,axis=1),axis=1),axis=1),axis=1),axis=1),axis=1),axis=1),axis=1)

      x_reconstruct, x_reconstruct_log_sigma_sq = model.reconstruct(sess,x_sample_nv) 
        
      scipy.io.savemat("results/mvae_final_completeloss_test4.mat",{"x_reconstruct":x_reconstruct,"x_reconstruct_log_sigma_sq":x_reconstruct_log_sigma_sq,"x_sample":x_sample_nv})

             