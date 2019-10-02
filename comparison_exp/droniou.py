from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.examples.tutorials.mnist import input_data
from my_nn_lib import Convolution2D, MaxPooling2D
from my_nn_lib import FullConnected, ReadOutLayer

import sys

logs_path = './logs/droniou'


print("Loading dataset...")

a = scipy.io.loadmat("../matlab/database/final_database_train.mat")
X_init = 1*a["final_database_train"]
#X_augm_train = X_init[0:1844,28:] #Only non, noisy data.
X_augm_train = X_init #all data
print(X_augm_train.shape)

#########################################################################################


        
class arch(object):
    def __init__(self, n_modalities, size_modalities, numModels, numFactors, numClass, numParam, used_modalities, batch_size=100,learning_rate=0.1, vanilla=False,
                                  corruption_level=0.3,
                                  softmaxnoise=1.0):
        self.vanilla=vanilla
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.n_modalities=n_modalities
        self.size_modalities=size_modalities
        self.size_input=np.sum(size_modalities)
        
        self.numModalities = size_modalities # nb of neurons as input for the modalities 
        self.numModels = numModels # nb of hidden neurons for the autoencoder on the modalities
	self.numFactors  = numFactors # nb of factor neurons for the modalities
        self.numParam = numParam # nb of neurons in the param layer
        self.numClass  = numClass # nb of neurons in the class layer

        self.corruption_level=corruption_level
        self.softmaxnoise=softmaxnoise
        
        self.corruption_level

        self.x = tf.placeholder(tf.float32, [None, self.size_input], name="input_x") #noiseless data
        self.used_modalities =  used_modalities
        print(self.used_modalities)
        # corrupt input with noise
        #print(self.x.get_shape())
        #sample = tf.where(tf.random_uniform([self.batch_size, self.size_input], minval=0,maxval=1) - (1.0-self.corruption_level) < 0,
        #                  tf.ones([self.batch_size, self.size_input]), tf.zeros([self.batch_size, self.size_input]))
        #self.x_corrupted = tf.multiply(sample, self.x)

        if(self.vanilla):
            print("vanilla approach")
            sample = tf.where(tf.random_uniform([self.batch_size, self.size_input], minval=0,maxval=1) - (1.0-self.corruption_level) < 0,
                              tf.ones([self.batch_size, self.size_input]), tf.zeros([self.batch_size, self.size_input]))
            self.x_corrupted = tf.multiply(sample, self.x) 
        else:
            self.x_corrupted =  tf.placeholder(tf.float32, [None, self.size_input], name="corrupt_x") #noisy data  


        
        self.modalities=self._slice_input(self.x,self.size_modalities)
        self.corrupted_modalities=self._slice_input(self.x_corrupted,self.size_modalities)

        self.create_variables()
        self.create_net()
        self.create_costs()
        
                         
        #self.saver = tf.train.Saver(tf.trainable_variables())
        self.saver = tf.train.Saver()


    def _slice_input(self, input_layer, size_mod):
	slices=[]
	count =0
	for i in range(len(size_mod)):
	    new_slice = tf.slice(input_layer, [0,count], [self.batch_size, size_mod[i]])
	    count+=size_mod[i]
	    slices.append(new_slice)
	return slices

        
    def create_variables(self):
        #To create:
        self.wModelClasses=[]
        self.wModalityModels=[]
        self.wModelFactors=[]
        self.wClassFactors=[]
        self.wFactorParams=[]
        self.bModels=[]
        self.bModalities=[]
        
        for i in range(len(self.modalities)):
            self.wModelClasses.append(tf.Variable(tf.truncated_normal([self.numModels[i], self.numClass], stddev=0.1), trainable=True))
            self.wModalityModels.append(tf.Variable(tf.truncated_normal([self.numModalities[i], self.numModels[i]], stddev=0.1), trainable=True))
            self.wModelFactors.append(tf.Variable(tf.truncated_normal([self.numModels[i], self.numFactors[i]], stddev=0.1), trainable=True))
            self.wClassFactors.append(tf.Variable(tf.truncated_normal([self.numClass, self.numFactors[i]], stddev=0.1), trainable=True))
            self.wFactorParams.append(tf.Variable(tf.truncated_normal([self.numFactors[i], self.numParam], stddev=0.1), trainable=True))

            self.bModels.append(tf.Variable(tf.zeros([self.numModels[i]]), trainable=True))
            self.bModalities.append(tf.Variable(tf.zeros([self.numModalities[i]]), trainable=True))

        self.bClass = tf.Variable(tf.zeros([self.numClass]), trainable=True)
        self.bParam = tf.Variable(tf.zeros([self.numParam]), trainable=True)
                                                                     
        # list of params which are updated by gradient descent (contains temporary variables, don't mind...)
        #self.params = [self.wModalityModels, self.wClassFactors]


        
    def create_costs(self):
        #self.params = self.layer.params
        #self._reconsModality1 = self.layer._modality1
        #self._reconsModality2 = self.layer._modality2
        #self._reconsModality3 = self.layer._modality3
        #self._class = self.layer._class

	# cost for the global reconstruction
        tmp_generativecost=[]
        for i in range(len(self.modalities)):
            tmp_generativecost.append(tf.reduce_mean(tf.reduce_sum(tf.square(self.modalities[i]-self._modalities[i] ), axis=1)))
            
        self._generativecost = tf.add_n(tmp_generativecost)

	# cost for the autoencoder reconstruction
        tmp_reconscost=[]
        for i in range(len(self.modalities)):
             tmp_reconscost.append( tf.reduce_mean(tf.reduce_sum(tf.square(self._reconsFromModels[i]-self.modalities[i] ), axis=1))  ) 
        self._reconscost = tf.add_n(tmp_reconscost)

	# define the currently used cost function (default values, will be modified)
	self._cost = self._generativecost

        train_vars_1=[]
        train_vars_1.extend(self.wModalityModels)
        train_vars_1.extend(self.bModalities)
        train_vars_1.extend(self.bModels)
        self.train_step_1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._reconscost, var_list=train_vars_1)
                        
        train_vars_2=[]
        train_vars_2.extend(self.wModelClasses)
        train_vars_2.extend(self.wModelFactors)
        train_vars_2.extend(self.wClassFactors)
        train_vars_2.extend(self.wFactorParams)
        train_vars_2.append(self.bClass)
        train_vars_2.append(self.bParam)
        self.train_step_2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._generativecost, var_list=train_vars_2)

        train_vars_3=[]
        train_vars_3.extend(self.wModalityModels)
        train_vars_3.extend(self.bModalities)
        train_vars_3.extend(self.bModels)
        train_vars_3.extend(self.wModelClasses)
        train_vars_3.append(self.wModelFactors)
        train_vars_3.append(self.wClassFactors)
        train_vars_3.append(self.wFactorParams)
        train_vars_3.append(self.bClass)
        train_vars_3.append(self.bParam)
                        
        self.train_step_3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._generativecost, var_list=train_vars_3)


    def train_1(self,sess, X):
	opt, cost = sess.run((self.train_step_1, self._reconscost ), feed_dict={self.x: X[:,28:], self.x_corrupted: X[:,:28]})
        #print(mod)
        #print(mod_rec)
	return cost

    def train_2(self,sess, X):
	opt, cost = sess.run((self.train_step_2, self._generativecost), feed_dict={self.x: X[:,28:], self.x_corrupted: X[:,:28]})
	return cost
    
    def train_3(self,sess, X):
	opt, cost = sess.run((self.train_step_3, self._generativecost), feed_dict={self.x: X[:,28:], self.x_corrupted: X[:,:28]})
	return cost


        
    def create_net(self):
        # autoencoder
        self._models=[]
        for i in range(len(self.modalities)):
            self._models.append(tf.nn.tanh( tf.matmul(self.corrupted_modalities[i], self.wModalityModels[i]) + self.bModels[i] ))

            
	# reconstruction from autoencoder (sigmoid for pictures coded between 0 and 1, linear for speech and trajectories)
        self._reconsFromModels=[]
        for i in range(len(self.modalities)):
             self._reconsFromModels.append( tf.nn.tanh( tf.matmul(self._models[i], tf.transpose(self.wModalityModels[i])) + self.bModalities[i]) )

        # class activation with same weighting for each modality (for visualization)
        tmp_models=[]
        for i in range(len(self.modalities)):
            tmp_models.append(tf.matmul(self._models[i], self.wModelClasses[i]))
        
        self._class = 1.0/len(self.modalities) * tf.nn.softmax(tf.add_n(tmp_models)+ self.bClass)



        # generate random weighting for each modality
        tmp_noise=[]
        for i in range(len(self.modalities)):
            tmp_noise.append(tf.random_uniform(shape=self._class.get_shape(),minval=0,maxval=1)* self.used_modalities[i])

            
            
        tmp_corupted=[]
        for i in range(len(self.modalities)):
            tmp_corupted.append(tf.multiply( tf.divide( tmp_noise[i],tf.add_n(tmp_noise)), tf.matmul(self._models[i],self.wModelClasses[i])))

                    
	# classification with random weighting + gaussian noise
        self._corruptedClass = tf.nn.softmax(tf.add_n(tmp_corupted) + self.bClass + tf.random_normal(shape=self._class.get_shape(), mean=0, stddev=self.softmaxnoise))


        self._reconsModelsFromClass=[]
        self._factorsClasses=[]
        self._factorsModels=[]
        self._factors=[]
        for i in range(len(self.modalities)):
	    # reconstruction of the autoencoder output from class layer
            self._reconsModelsFromClass.append(tf.nn.tanh( tf.matmul(self._corruptedClass, tf.transpose(self.wModelClasses[i])) + self.bModels[i] ))
	    # projection of class layer onto factor layer
	    self._factorsClasses.append(tf.matmul( self._corruptedClass, self.wClassFactors[i] ))
            # projection of autoencoder output
	    self._factorsModels.append(tf.matmul( self._models[i], self.wModelFactors[i] ))
            # factor layers values
	    self._factors.append(tf.multiply(self._factorsClasses[i], self._factorsModels[i]))


        
	# param layer
        tmp_weighted_factors=[]
        for i in range(len(self.modalities)):
            tmp_weighted_factors.append(tf.matmul(self._factors[i], self.wFactorParams[i] ))

        self._param = 1.0/len(self.modalities) * tf.nn.softplus(tf.add_n(tmp_weighted_factors)+ self.bParam)

        
            
        # generate random weighting for each modality
        tmp_noise_bis=[]
        for i in range(len(self.modalities)):
            tmp_noise_bis.append( tf.random_uniform(shape=self._param.get_shape(),minval=0,maxval=1)* self.used_modalities[i])
            
        tmp_corrupted_params=[]
        for i in range(len(self.modalities)):
            tmp_corrupted_params.append(tf.multiply(tf.multiply( tmp_noise_bis[i],tf.add_n(tmp_noise_bis)),tf.matmul(self._factors[i],self.wFactorParams[i] )))

        # classification with random weighting + gaussian noise
        self._corruptedParam = tf.nn.softmax(tf.add_n(tmp_corrupted_params) + self.bParam)

        self._reconsModels=[]
        self._modalities=[]
        for i in range(len(self.modalities)):
	    # reconstruction of the autoencoder output using the gated network
            self._reconsModels.append( tf.nn.tanh( tf.matmul( tf.multiply(self._factorsClasses[i], tf.matmul(self._corruptedParam, tf.transpose(self.wFactorParams[i]))), tf.transpose(self.wModelFactors[i])) + self.bModels[i] ))
            # global reconstruction of each modality
	    self._modalities.append( tf.nn.tanh( tf.matmul( self._reconsModels[i], tf.transpose(self.wModalityModels[i]) ) + self.bModalities[i]))

        self.reconst_modalities = tf.concat(self._modalities,1)


    def reconstruct(self,sess, X_test):
        """ Use network to reconstruct given data. """
        x_rec = sess.run((self.reconst_modalities),
                                                 feed_dict={self.x_corrupted: X_test})
        return x_rec


                                                            









            
def shuffle_data(x):
    y = list(x)
    np.random.shuffle(y)
    return np.asarray(y)

        





def train_whole(sess, network, input_data, learning_rate=0.0001, batch_size=100, training_epochs=10, display_step=100):
	
    # Write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

	
    # Training cycle for whole network
    for epoch in range(training_epochs):
	avg_cost = 0.
	total_batch = int(n_samples / batch_size)
	X_shuffled = shuffle_data(input_data)

	# Loop over all batches
	for i in range(total_batch):
            batch_xs_augmented = X_shuffled[batch_size*i:batch_size*i+batch_size] 
	    #batch_xs   = np.asarray([item   for item in batch_xs_augmented])
            batch_xs=batch_xs_augmented
	    
	    # Fit training using batch data
            if(epoch<4500):
	        cost = network.train_1(sess, batch_xs)
            elif(epoch<9000):
                cost = network.train_2(sess, batch_xs)
            else:
                cost = network.train_3(sess, batch_xs)
                
            
            
	    avg_cost += cost / n_samples * batch_size

	# Display logs per epoch step
	if epoch % display_step == 0:
	    print("Epoch: %04d / %04d, Cost= %04f" % \
		  (epoch,training_epochs,avg_cost))
    

            
	if epoch % display_step*5 == 0:
            save_path = network.saver.save(sess, "./models/droniou_temp.ckpt")
                        
    	
    save_path = network.saver.save(sess, "./models/droniou_complete.ckpt")
                        

















            
if __name__ == '__main__':

    vanilla=False
    if(len(sys.argv)<2):
        print('please provide one argument: 1 training with only complete data, 0 training with Martinas method')
        exit()
    test_id= int(sys.argv[1:][0])
    if(test_id==1):
        print('VANILLA: training with complete data only')
        vanilla=True
    else:
        print('MARTINA: training with martinas approach')
                                     

    if vanilla:
        X_augm_train=X_augm_train[0:1845,:]
    n_samples = X_augm_train.shape[0]    
    learning_rate = 0.0001
    batch_size = 100

    # Train Network
    print('Train net')

    sess = tf.InteractiveSession()
    n_modalities=10
    size_modalities=[4,4,4,4,1,1,1,1,4,4]
    numModels=[10,10,10,10,2,2,2,2,10,10]
    numFactors=[10,10,10,10,2,2,2,2,10,10]
    used_modalities= [1,1,  1,1,  1,1,  1,1,  1,1]
    numClass=1
    numParam=100
    corruption_level=0.3
    softmaxnoise=1.0
    network = arch( n_modalities, size_modalities, numModels, numFactors, numClass, numParam, used_modalities, batch_size,learning_rate,vanilla,corruption_level,
                                      softmaxnoise)
    init = tf.global_variables_initializer() #tf.initialize_all_variables() #
    sess.run(init)
    
	
    train_whole(sess,network, X_augm_train, training_epochs=15000,batch_size=batch_size)
