#loadmodel.py

import numpy as np
import tensorflow as tf
import scipy.io
import math
import sys
sys.path.append('../../comparison_exp/')


#*****************************#
from droniou import arch

#*****************************#

print("Loading dataset...")

# a = scipy.io.loadmat("matlab/database/n_database.mat")
a = scipy.io.loadmat("../matlab/database/final_database_test.mat")
X_init = 1*a["final_database_test"]
X_augm_test = X_init
print(X_augm_test.shape)
sample_init = 100

def test_1():
	# Test 1: complete data
	print('Test 1')
	x_sample = X_augm_test[sample_init:sample_init+batch_size,:28]  
	x_reconstruct = network.reconstruct(sess, x_sample)
	scipy.io.savemat("results/mvae_final_test1.mat",{"x_reconstruct":x_reconstruct,"x_sample":x_sample})
        
	################################################################################################################

def test_2():
 	# Test 2: vision only
 	print('Test 2')
        x_sample = X_augm_test[sample_init:sample_init+batch_size,:28]
	x_sample_nv_1 = np.full((x_sample.shape[0],8),-2)  
        x_sample_nv_2 = X_augm_test[sample_init:sample_init+batch_size,8:16] 
 	x_sample_nv_3 = np.full((x_sample.shape[0],12),-2)  
 	x_sample_nv = np.append( x_sample_nv_1, np.append( x_sample_nv_2, x_sample_nv_3, axis=1), axis=1)
                 
        x_reconstruct = network.reconstruct(sess, x_sample_nv)
        scipy.io.savemat("results/mvae_final_test2.mat",{"x_reconstruct":x_reconstruct,"x_sample":x_sample_nv})
 
def test_3():
 	# Test 3: joint (t) and vision -- for control loop
 	print('Test 3')
        x_sample = X_augm_test[sample_init:sample_init+batch_size,:28]
        x_sample_nv_1 = np.full((x_sample.shape[0],4),-2)
        x_sample_nv_2 = X_augm_test[sample_init:sample_init+batch_size,4:16]
        x_sample_nv_3 = np.full((x_sample.shape[0],12),-2)
        x_sample_nv = np.append( x_sample_nv_1, np.append( x_sample_nv_2, x_sample_nv_3, axis=1), axis=1)
        
 	x_reconstruct = network.reconstruct(sess, x_sample_nv)
        scipy.io.savemat("results/mvae_final_test3.mat",{"x_reconstruct":x_reconstruct,"x_sample":x_sample_nv})
 		         

def test_4():
        # Test 4: only data at time t -- for prediction ?
 	print('Test 4')
        x_sample = X_augm_test[sample_init:sample_init+batch_size,:28]
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
        
        x_reconstruct = network.reconstruct(sess, x_sample_nv)
        scipy.io.savemat("results/mvae_final_test4.mat",{"x_reconstruct":x_reconstruct,"x_sample":x_sample_nv})
                         
 
 	#	# Test 5: use VAE for prediction: first reconstruct from vision only, then reconstruct again from reconstructed data
        #        print('USING VAE FOR PREDICTIONS')
        #        x_sample_nv_1 = np.full((x_sample.shape[0],8),-2)
        #        x_sample_nv_2 = X_augm_test[sample_init:sample_init+batch_size,8:16]
        #        x_sample_nv_3 = np.full((x_sample.shape[0],12),-2)
        #        x_sample_nv = np.append( x_sample_nv_1, np.append( x_sample_nv_2, x_sample_nv_3, axis=1), axis=1)
        #
        #        x_reconstruct = network.reconstruct(sess, x_sample_nv)
        #        #HERE WE HAVE A PROBABILITY DISTRIBUTION...SHALL WE SAMPLE?
 	#	x_sample_nv_1 = np.full((x_sample.shape[0],4),-2)  
 	#	x_sample_nv_2 = x_reconstruct[:,0:4] 
 	#	x_sample_nv_3 = np.full((x_sample.shape[0],4),-2)  
 	#	x_sample_nv_4 = x_reconstruct[:,8:12] 
 	#	x_sample_nv_5 = np.full((x_sample.shape[0],1),-2)  
 	#	x_sample_nv_6 = x_reconstruct[:,16:17] 
 	#	x_sample_nv_7 = np.full((x_sample.shape[0],1),-2)  
 	#	x_sample_nv_8 = x_reconstruct[:,18:19]
        #        x_sample_nv_9 = np.full((x_sample.shape[0],4),-2)
        #        x_sample_nv_10 = x_reconstruct[:,20:24]
        #        x_sample_nv = np.append(x_sample_nv_1,
        #                                np.append(x_sample_nv_2,
        #                                           np.append(x_sample_nv_3,
        #                                                     np.append(x_sample_nv_4,
        #                                                               np.append(x_sample_nv_5,
        #                                                                         np.append(x_sample_nv_6,
        #                                                                                   np.append(x_sample_nv_7,
        #                                                                                             np.append(x_sample_nv_8,
        #                                                                                                       np.append(x_sample_nv_9,
        #                                                                                                                 x_sample_nv_10,axis=1)
        #                                                                                                       ,axis=1),axis=1),axis=1),axis=1),axis=1),axis=1),axis=1),axis=1)
        #        
        #        x_reconstruct = network.reconstruct(sess, x_sample_nv)
        #        scipy.io.savemat("results/mvae_final_test5.mat",{"x_reconstruct":x_reconstruct,"x_sample":x_sample_nv})

 
 		                                                                
if __name__ == '__main__':
        
        test_id= int(sys.argv[1:][0])
        print("test_id:" + str(test_id))

        ################################################################################################################
        with tf.Graph().as_default() as g:
	        with tf.Session() as sess:
                
		        # Network parameters
                        
                        learning_rate = 0.0001
                        batch_size = 1000
                        n_modalities=5
                        size_modalities=[4,4,4,4,1,1,1,1,4,4]
                        numModels=[10,10,10,10,2,2,2,2,10,10]
                        numFactors=[10,10,10,10,2,2,2,2,10,10]
                        options = {1 : [1,1,  1,1,  1,1,  1,1,  1,1],
                                   2 : [0,0,  1,1,  0,0,  0,0,  0,0],
                                   3 : [0,1,  1,1,  0,0,  0,0,  0,0],
                                   4 : [0,1,  0,1,  0,1,  0,1,  0,1]}
                        used_modalities=options[test_id]
                        corruption_level=0.0
                        softmaxnoise=0.0
                        numClass=1
                        numParam=100
                        vanilla=True
                        
                        network = arch( n_modalities, size_modalities, numModels, numFactors, numClass, numParam, used_modalities, batch_size,learning_rate,vanilla,corruption_level, softmaxnoise)
                

		        new_saver = tf.train.Saver() 
                        new_saver.restore(sess, "./models/droniou_complete.ckpt")
		        print("Model restored.")
						
                        if(test_id==1):
                                test_1()
                        if(test_id==2):
                                test_2()
                        if(test_id==3):
                                test_3()
                        if(test_id==4):
                                test_4()
                        
