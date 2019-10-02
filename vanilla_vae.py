import numpy as np
import tensorflow as tf
import scipy.io
import math
import sys

import matplotlib.pyplot as plt


logs_path = './logs/vanilla_vae'


print("Loading dataset...")

a = scipy.io.loadmat("matlab/database/final_database_train.mat")
X_init = 1*a["final_database_train"]
X_augm_train = X_init#[:1845,:] # #np.append(X_train_all,X_train_no_mc,axis=0)
##########################################################################################


def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(1.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(1.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),  minval=low, maxval=high,  dtype=tf.float64)
    #stddev = np.sqrt(1.0 / (fan_in + fan_out))
    #return tf.random_normal((fan_in, fan_out), mean = 0.0, stddev=stddev, dtype=tf.float64)

class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, sess, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        n_input = network_architecture["n_input"]
        
        # tf Graph input
        self.x = tf.placeholder(tf.float64, [None, n_input])
        self.x_noiseless = tf.placeholder(tf.float64, [None, n_input])

        # corrupt input with noise
        corruption_level = 0.3
        sample = tf.where(tf.random_uniform([batch_size, n_input], minval=0,maxval=1, dtype=tf.float64) - (1.0-corruption_level) < 0,
                          tf.ones([batch_size, n_input], dtype=tf.float64), tf.zeros([batch_size, n_input], dtype=tf.float64))
        self.x_corrupted = tf.multiply(sample, self.x)

        
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = sess
        self.sess.run(init)

        self.saver = tf.train.Saver()
    
    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"], 
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        self.n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, self.n_z), 0, 1, 
                               dtype=tf.float64)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean, self.x_reconstr_log_sigma_sq = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
            
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1), dtype=tf.float64),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2), dtype=tf.float64),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z), dtype=tf.float64),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z), dtype=tf.float64)}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float64)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float64)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float64)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float64))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1), dtype=tf.float64),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2), dtype=tf.float64),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input), dtype=tf.float64),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input), dtype=tf.float64)}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float64)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float64)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float64)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float64))}
        return all_weights
            
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x_corrupted, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        x_reconstr_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        x_reconstr_log_sigma_sq = tf.log(tf.exp(tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])) )

        return (x_reconstr_mean, x_reconstr_log_sigma_sq)
            
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        self.reconstr_loss = \
            0.5 * tf.reduce_sum(tf.square(self.x_noiseless - self.x_reconstr_mean) / tf.exp(self.x_reconstr_log_sigma_sq),1) \
            + 0.5 * tf.reduce_sum(self.x_reconstr_log_sigma_sq,1) \
            + 0.5 * self.n_z/2 * np.log(2*math.pi) 
        self.reconstr_loss = tf.reduce_mean(self.reconstr_loss)

        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        self.latent_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1))
        self.cost = tf.reduce_mean(self.reconstr_loss + self.latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        


    def partial_fit(self,sess, X, X_noiseless, epoch):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """

        opt, cost, recon, latent, x_rec = sess.run((self.optimizer, self.cost, self.reconstr_loss, self.latent_loss, self.x_reconstr_mean), 
            feed_dict={self.x: X, self.x_noiseless: X_noiseless})
        return cost, recon, latent, x_rec
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})
    
    def reconstruct(self,sess, X_test):
        """ Use VAE to reconstruct given data. """
        x_rec_mean,x_rec_log_sigma_sq = sess.run((self.x_reconstr_mean, self.x_reconstr_log_sigma_sq), 
            feed_dict={self.x: X_test})
        return x_rec_mean, x_rec_log_sigma_sq

            
def shuffle_data(x):
    y = list(x)
    np.random.shuffle(y)
    return np.asarray(y)



def train_whole(sess, vae, input_data, learning_rate=0.0001, batch_size=100, training_epochs=10, display_step=100):
    
    # Write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    
    # Training cycle for whole network
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_recon = 0.
        avg_latent = 0.
        
        total_batch = int(n_samples / batch_size)

        X_shuffled = shuffle_data(input_data)

        # Loop over all batches
        for i in range(total_batch):


            batch_xs_augmented = X_shuffled[batch_size*i:batch_size*i+batch_size] 
            
            batch_xs   = np.asarray([item[:28]   for item in batch_xs_augmented])#np.asarray([item[:18]   for item in batch_xs_augmented])
            batch_xs_noiseless   = np.asarray([item[28:]   for item in batch_xs_augmented])#np.asarray([item[:18]   for item in batch_xs_augmented])
                        # batch_xs_noiseless_J   = np.asarray([item[8:12]   for item in batch_xs_noiseless])

            
            # Fit training using batch data
            cost, recon, latent, x_rec = vae.partial_fit(sess, batch_xs, batch_xs_noiseless,epoch)
            avg_cost += cost / n_samples * batch_size
            avg_recon += recon / n_samples * batch_size
            avg_latent += latent / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch: %04d / %04d, Cost= %04f, Recon= %04f, Latent= %04f" % \
                (epoch,training_epochs,avg_cost, avg_recon, avg_latent))



    save_path = vae.saver.save(vae.sess, "./models/vanilla_vae_compl_data.ckpt")
    



if __name__ == '__main__':

    if(len(sys.argv)<2):
        print('please provide one argument: 1 training with only complete data, 0 training with Martinas method')
        exit()

    test_id= int(sys.argv[1:][0])
    if(test_id==1):
        print('training with complete data only')
        X_augm_train=X_augm_train[0:1845,:]
    else:
        print('training with martinas approach')
    print(X_augm_train.shape)
    n_samples = X_augm_train.shape[0]
    
    learning_rate = 0.001
    batch_size = 100

    # Train Network
    print('Train net')

    sess = tf.InteractiveSession()

    network_architecture = \
    dict(n_hidden_recog_1=100, # 1st layer encoder neurons
         n_hidden_recog_2=100, # 2nd layer encoder neurons
         n_hidden_gener_1=100, # 1st layer decoder neurons
         n_hidden_gener_2=100, # 2nd layer decoder neurons
         n_input=28, # MNIST data input (img shape: 28*28)
         n_z=28)  # dimensionality of latent space
        
    vae = VariationalAutoencoder(sess,network_architecture,  learning_rate=learning_rate,  batch_size=batch_size)
    # vae.print_layers_size()
    
    train_whole(sess,vae, X_augm_train, training_epochs=8000,batch_size=batch_size)

    
