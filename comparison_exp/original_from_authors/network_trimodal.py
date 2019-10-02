import numpy
import pylab
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

SMALL = 0.000001

class NetworkLayer(object):
    """ 
        Network definition
    """

    def __init__(self, inputs, numModality1, numModel1, numFactor1, numModality2, numModel2, numFactor2, numModality3, numModel3, numFactor3, numClass, numParam,
                 corruption_level=0.0, 
                 numpy_rng=None, theano_rng=None, softmaxnoise=1.0):
        
	self.numModality1 = numModality1 # nb of neurons as input for first modality 
        self.numModel1 = numModel1 # nb of hidden neurons for the autoencoder on modality 1
	self.numFactor1  = numFactor1 # nb of factor neurons for modality 1
        self.numModality2 = numModality2
        self.numModel2 = numModel2
        self.numFactor2  = numFactor2
        self.numModality3 = numModality3
        self.numModel3 = numModel3
        self.numFactor3  = numFactor3
        self.numParam = numParam # nb of neurons in the param layer
        self.numClass  = numClass # nb of neurons in the class layer


        self.corruption_level = corruption_level # % of input neurons set to 0
       
       	# define learning rate modifiers to update biases more slowly than weights
        self.learningrate_modifiers = {}
	# wXYn denotes the matrix connecting layer X to layer Y for modality n
        self.learningrate_modifiers['wModalityModel1']     = 1.0
        self.learningrate_modifiers['wModelClass1']     = 1.0
        self.learningrate_modifiers['wModelFactor1']     = 1.0
        self.learningrate_modifiers['wClassFactor1']     = 1.0
        self.learningrate_modifiers['wFactorParam1']     = 1.0
        self.learningrate_modifiers['wModalityModel2']     = 1.0
        self.learningrate_modifiers['wModelClass2']     = 1.0
        self.learningrate_modifiers['wModelFactor2']     = 1.0
        self.learningrate_modifiers['wClassFactor2']     = 1.0
        self.learningrate_modifiers['wFactorParam2']     = 1.0
        self.learningrate_modifiers['wModalityModel3']     = 1.0
        self.learningrate_modifiers['wModelClass3']     = 1.0
        self.learningrate_modifiers['wModelFactor3']     = 1.0
        self.learningrate_modifiers['wClassFactor3']     = 1.0
        self.learningrate_modifiers['wFactorParam3']     = 1.0
        
	# biases
        self.learningrate_modifiers['bModel1']    = 0.1
        self.learningrate_modifiers['bModel2']   = 0.1
        self.learningrate_modifiers['bModel3']   = 0.1
        self.learningrate_modifiers['bModality1']   = 0.1
        self.learningrate_modifiers['bModality2']   = 0.1
        self.learningrate_modifiers['bModality3']   = 0.1
        self.learningrate_modifiers['bParam']    = 0.1
        self.learningrate_modifiers['bClass']   = 0.1


	# random generator for initialisation
        if not numpy_rng:  
            numpy_rng = numpy.random.RandomState(1)

	# random generator for training noise
        if not theano_rng:  
            theano_rng = RandomStreams(1)

        # SET UP VARIABLES AND PARAMETERS 
        # random values for matrices
        wModalityModel1_init = numpy_rng.normal(size=(numModality1, numModel1)).astype(theano.config.floatX)*0.01
        wModalityModel2_init = numpy_rng.normal(size=(numModality2, numModel2)).astype(theano.config.floatX)*0.01
        wModalityModel3_init = numpy_rng.normal(size=(numModality3, numModel3)).astype(theano.config.floatX)*0.01
        wModelClass1_init = numpy_rng.normal(size=(numModel1, numClass)).astype(theano.config.floatX)*0.01
        wModelClass2_init = numpy_rng.normal(size=(numModel2, numClass)).astype(theano.config.floatX)*0.01
        wModelClass3_init = numpy_rng.normal(size=(numModel3, numClass)).astype(theano.config.floatX)*0.01
        wModelFactor1_init = numpy_rng.normal(size=(numModel1, numFactor1)).astype(theano.config.floatX)*0.01
        wModelFactor2_init = numpy_rng.normal(size=(numModel2, numFactor2)).astype(theano.config.floatX)*0.01
        wModelFactor3_init = numpy_rng.normal(size=(numModel3, numFactor3)).astype(theano.config.floatX)*0.01
        wClassFactor1_init = numpy_rng.normal(size=(numClass, numFactor1)).astype(theano.config.floatX)*0.01
        wClassFactor2_init = numpy_rng.normal(size=(numClass, numFactor2)).astype(theano.config.floatX)*0.01
        wClassFactor3_init = numpy_rng.normal(size=(numClass, numFactor3)).astype(theano.config.floatX)*0.01
        wFactorParam1_init = numpy_rng.normal(size=(numFactor1, numParam)).astype(theano.config.floatX)*0.01
        wFactorParam2_init = numpy_rng.normal(size=(numFactor2, numParam)).astype(theano.config.floatX)*0.01
        wFactorParam3_init = numpy_rng.normal(size=(numFactor3, numParam)).astype(theano.config.floatX)*0.01

	# send matrices to theano
        self.wModalityModel1 = theano.shared(value = wModalityModel1_init, name = 'wModalityModel1')
        self.wModalityModel2 = theano.shared(value = wModalityModel2_init, name = 'wModalityModel2')
        self.wModalityModel3 = theano.shared(value = wModalityModel3_init, name = 'wModalityModel3')
        self.wModelClass1 = theano.shared(value = wModelClass1_init, name = 'wModelClass1')
        self.wModelClass2 = theano.shared(value = wModelClass2_init, name = 'wModelClass2')
        self.wModelClass3 = theano.shared(value = wModelClass3_init, name = 'wModelClass3')
        self.wModelFactor1 = theano.shared(value = wModelFactor1_init, name = 'wModelFactor1')
        self.wModelFactor2 = theano.shared(value = wModelFactor2_init, name = 'wModelFactor2')
        self.wModelFactor3 = theano.shared(value = wModelFactor3_init, name = 'wModelFactor3')
        self.wClassFactor1 = theano.shared(value = wClassFactor1_init, name = 'wClassFactor1')
        self.wClassFactor2 = theano.shared(value = wClassFactor2_init, name = 'wClassFactor2')
        self.wClassFactor3 = theano.shared(value = wClassFactor3_init, name = 'wClassFactor3')
        self.wFactorParam1 = theano.shared(value = wFactorParam1_init, name = 'wFactorParam1')
        self.wFactorParam2 = theano.shared(value = wFactorParam2_init, name = 'wFactorParam2')
        self.wFactorParam3 = theano.shared(value = wFactorParam3_init, name = 'wFactorParam3')

	#biases
        self.bModel1 = theano.shared(value = numpy.zeros(numModel1, dtype=theano.config.floatX), name='bModel1')
        self.bModel2 = theano.shared(value = numpy.zeros(numModel2, dtype=theano.config.floatX), name='bModel2')
        self.bModel3 = theano.shared(value = numpy.zeros(numModel3, dtype=theano.config.floatX), name='bModel3')
        self.bClass= theano.shared(value = numpy.zeros(numClass, dtype=theano.config.floatX), name='bClass')
        self.bParam = theano.shared(value = numpy.zeros(numParam, dtype=theano.config.floatX), name='bParam')
        self.bModality1 = theano.shared(value = numpy.zeros(numModality1, dtype=theano.config.floatX), name='bModality1')
        self.bModality2 = theano.shared(value = numpy.zeros(numModality2, dtype=theano.config.floatX), name='bModality2')
        self.bModality3 = theano.shared(value = numpy.zeros(numModality3, dtype=theano.config.floatX), name='bModality3')
        
        self.inputs = inputs

	# list of params which are updated by gradient descent (contains temporary variables, don't mind...)
        self.params = [self.wModalityModel1, self.wModalityModel2]

        #=======================
	# NETWORK DESCRIPTION
	#=======================

	# inputs consist in one line for each training sample, where all modalities have been concatenated.
	# need to split each modality
	self.modality1 = self.inputs[:, :numModality1]
        self.modality2 = self.inputs[:, numModality1:numModality1+numModality2]
        self.modality3 = self.inputs[:, numModality1+numModality2:]
        
	# corrupt input with noise
        self._corruptedModality1 = theano_rng.binomial(size=self.modality1.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.modality1
        self._corruptedModality2 = theano_rng.binomial(size=self.modality2.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.modality2
        self._corruptedModality3 = theano_rng.binomial(size=self.modality3.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.modality3

        # autoencoder
        self._model1 = T.nnet.sigmoid( T.dot(self._corruptedModality1, self.wModalityModel1) + self.bModel1 )
        self._model2 = T.nnet.sigmoid( T.dot(self._corruptedModality2, self.wModalityModel2) + self.bModel2 )
        self._model3 = T.nnet.sigmoid( T.dot(self._corruptedModality3, self.wModalityModel3) + self.bModel3 )
        
	# reconstruction from autoencoder (sigmoid for pictures coded between 0 and 1, linear for speech and trajectories)
	self._reconsFromModel1 =  T.nnet.sigmoid( T.dot(self._model1, self.wModalityModel1.T) + self.bModality1)
        self._reconsFromModel2 =  ( T.dot(self._model2, self.wModalityModel2.T) + self.bModality2)
        self._reconsFromModel3 =  ( T.dot(self._model3, self.wModalityModel3.T) + self.bModality3)

	# class activation with same weighting for each modality (for visualization)
        self._class = T.nnet.softmax(( 0.33*T.dot(self._model1, self.wModelClass1) + 0.33*T.dot(self._model2, self.wModelClass2)  + 0.33*T.dot(self._model3, self.wModelClass3)+ self.bClass))


	# generate random weighting for each modality
        self._random1 = theano_rng.uniform(size=self._class.shape,low=0,high=1)
        self._random2 = theano_rng.uniform(size=self._class.shape,low=0,high=1)
        self._random3 = theano_rng.uniform(size=self._class.shape,low=0,high=1)
        total=self._random1+self._random2+self._random3
        self._random12 = self._random1/total
        self._random22 = self._random2/total
        self._random32 = self._random3/total

	# classification with random weighting + gaussian noise
        self._corruptedClass = T.nnet.softmax(( self._random12*T.dot(self._model1, self.wModelClass1) + self._random22*T.dot(self._model2, self.wModelClass2) + self._random32*T.dot(self._model3, self.wModelClass3) + self.bClass) + theano_rng.normal(size=self._class.shape, avg=0, std=softmaxnoise, dtype=theano.config.floatX))

	# reconstruction of the autoencoder output from class layer
        self._reconsModel1FromClass = T.nnet.sigmoid( T.dot(self._corruptedClass, self.wModelClass1.T) + self.bModel1 )
        self._reconsModel2FromClass = T.nnet.sigmoid( T.dot(self._corruptedClass, self.wModelClass2.T) + self.bModel2 )
        self._reconsModel3FromClass = T.nnet.sigmoid( T.dot(self._corruptedClass, self.wModelClass3.T) + self.bModel3 )

	# some values used to analyse the learned classification, when some modalities are missing
        self._classFrom1 = T.nnet.softmax( T.dot(self._model1, self.wModelClass1) + self.bClass)
        self._classFrom2 = T.nnet.softmax( T.dot(self._model2, self.wModelClass2) + self.bClass)
        self._classFrom3 = T.nnet.softmax( T.dot(self._model3, self.wModelClass3) + self.bClass)
        self._classFrom12 = T.nnet.softmax( 0.5*T.dot(self._model1, self.wModelClass1) + 0.5*T.dot(self._model2, self.wModelClass2) + self.bClass)
        self._classFrom13 = T.nnet.softmax( 0.5*T.dot(self._model1, self.wModelClass1) + 0.5*T.dot(self._model3, self.wModelClass3) + self.bClass)
        self._classFrom23 = T.nnet.softmax( 0.5*T.dot(self._model2, self.wModelClass2) + 0.5*T.dot(self._model3, self.wModelClass3) + self.bClass)
        
	# projection of class layer onto factor layer
	self._factorsClass1 = T.dot( self._corruptedClass, self.wClassFactor1 )
        self._factorsClass2 = T.dot( self._corruptedClass, self.wClassFactor2 )
        self._factorsClass3 = T.dot( self._corruptedClass, self.wClassFactor3 )
        # projection of autoencoder output
	self._factorsModel1 = T.dot( self._model1, self.wModelFactor1 )
        self._factorsModel2 = T.dot( self._model2, self.wModelFactor2 )
        self._factorsModel3 = T.dot( self._model3, self.wModelFactor3 )
        # factor layers values
	self._factors1 = self._factorsClass1 * self._factorsModel1
        self._factors2 = self._factorsClass2 * self._factorsModel2
        self._factors3 = self._factorsClass3 * self._factorsModel3
        
	# param layer
	self._param = T.nnet.softplus( 0.33*T.dot(self._factors1, self.wFactorParam1) + 0.33*T.dot(self._factors2, self.wFactorParam2) + 0.33*T.dot(self._factors3, self.wFactorParam3) + self.bParam)
        # for analysis, when some modalities are missing
	self._paramFrom1 = T.nnet.softplus( T.dot(self._factors1, self.wFactorParam1) + self.bParam)
        self._paramFrom2 = T.nnet.softplus( T.dot(self._factors2, self.wFactorParam2) + self.bParam)
        self._paramFrom3 = T.nnet.softplus( T.dot(self._factors3, self.wFactorParam3) + self.bParam)
        self._paramFrom12 = T.nnet.softplus( 0.5*T.dot(self._factors1, self.wFactorParam1) + 0.5*T.dot(self._factors2, self.wFactorParam2) + self.bParam)
        self._paramFrom13 = T.nnet.softplus( 0.5*T.dot(self._factors1, self.wFactorParam1) + 0.5*T.dot(self._factors3, self.wFactorParam3) + self.bParam)
        self._paramFrom23 = T.nnet.softplus( 0.5*T.dot(self._factors2, self.wFactorParam2) + 0.5*T.dot(self._factors3, self.wFactorParam3) + self.bParam)

	# param layer with random weighting
        self._randombis1 = theano_rng.uniform(size=self._param.shape,low=0,high=1)
        self._randombis2 = theano_rng.uniform(size=self._param.shape,low=0,high=1)
        self._randombis3 = theano_rng.uniform(size=self._param.shape,low=0,high=1)
        total2=self._randombis1+self._randombis2+self._randombis3
        self._randombis12 = self._randombis1/total2
        self._randombis22 = self._randombis2/total2
        self._randombis32 = self._randombis3/total2
        self._corruptedParam = T.nnet.softplus( self._randombis12*T.dot(self._factors1, self.wFactorParam1) + self._randombis22*T.dot(self._factors2, self.wFactorParam2) + self._randombis32*T.dot(self._factors3, self.wFactorParam3) + self.bParam) 

	# reconstruction of the autoencoder output using the gated network
        self._reconsModel1 = T.nnet.sigmoid( T.dot(self._factorsClass1 * T.dot(self._corruptedParam, self.wFactorParam1.T), self.wModelFactor1.T) + self.bModel1)
        self._reconsModel2 = T.nnet.sigmoid( T.dot(self._factorsClass2 * T.dot(self._corruptedParam, self.wFactorParam2.T), self.wModelFactor2.T) + self.bModel2)
        self._reconsModel3 = T.nnet.sigmoid( T.dot(self._factorsClass3 * T.dot(self._corruptedParam, self.wFactorParam3.T), self.wModelFactor3.T) + self.bModel3)
        # global reconstruction of each modality
	self._modality1 = T.nnet.sigmoid( T.dot( self._reconsModel1, self.wModalityModel1.T ) + self.bModality1)
        self._modality2 = ( T.dot( self._reconsModel2, self.wModalityModel2.T ) + self.bModality2)
        self._modality3 = ( T.dot( self._reconsModel3, self.wModalityModel3.T ) + self.bModality3)


        # COMPILE SOME FUNCTIONS THAT MAY BE USEFUL LATER FOR VISUALIZATION
        self.corruptedModality1 = theano.function([self.inputs], self._corruptedModality1)
        self.corruptedModality2 = theano.function([self.inputs], self._corruptedModality2)
        self.classes = theano.function([self.inputs], self._class)
        self.reconsModality1 = theano.function([self.inputs], self._modality1)
        self.reconsModality2 = theano.function([self.inputs], self._modality2)
        self.reconsModality3 = theano.function([self.inputs], self._modality3)
        self.parameters = theano.function([self.inputs], self._param)
        self.classFrom1 = theano.function([self._corruptedModality1], self._classFrom1)
        self.classFrom2 = theano.function([self._corruptedModality2], self._classFrom2)
        self.classFrom3 = theano.function([self._corruptedModality3], self._classFrom3)

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])


class Network(object):
    """ 
	Wrap the network with cost functions and gradients
    """
    def __init__(self, numModality1, numModel1, numFactor1, numModality2, numModel2, numFactor2, numModality3, numModel3, numFactor3, numClass, numParam,
                 corruption_level=0.0, 
                 numpy_rng=None, theano_rng=None, inputs=None, softmaxnoise=1.0):

        self.numModality1 = numModality1
        self.numModel1 = numModel1
        self.numParam = numParam
        self.numFactor1  = numFactor1
        self.numClass  = numClass
        self.numModality2 = numModality2
        self.numModel2 = numModel2
        self.numFactor2  = numFactor2
        self.numModality3 = numModality3
        self.numModel3 = numModel3
        self.numFactor3  = numFactor3
        self.corruption_level = corruption_level


        if inputs is None:
            self.inputs = T.matrix(name = 'inputs') 
        else:
            self.inputs = inputs

        if not numpy_rng:  
            numpy_rng = numpy.random.RandomState(1)

        if not theano_rng:  
            theano_rng = RandomStreams(1)

        # SET UP MODEL USING LAYER
        self.layer = NetworkLayer(self.inputs, numModality1, numModel1, numFactor1, numModality2, numModel2, numFactor2, numModality3, numModel3, numFactor3, numClass, numParam,
                                               corruption_level=corruption_level,
                                               numpy_rng=numpy_rng, theano_rng=theano_rng, softmaxnoise=softmaxnoise)
        self.params = self.layer.params
        self._reconsModality1 = self.layer._modality1
        self._reconsModality2 = self.layer._modality2
        self._reconsModality3 = self.layer._modality3
        self._class = self.layer._class

	# cost for the global reconstruction
        self._generativecost = T.mean(T.sum(((self.layer.modality1-self._reconsModality1)**2),axis=1)) + T.mean(T.sum(((self.layer.modality2-self._reconsModality2)**2),axis=1)) + T.mean(T.sum(((self.layer.modality3-self._reconsModality3)**2),axis=1))    
	#cost for the autoencoder reconstruction
        self._reconscost = T.mean(T.sum(((self.layer._reconsFromModel1-self.layer.modality1)**2),axis=1)) + T.mean(T.sum(((self.layer._reconsFromModel2 - self.layer.modality2)**2),axis=1)) + T.mean(T.sum(((self.layer._reconsFromModel3 - self.layer.modality3)**2),axis=1))   
        
	# define the currently used cost function (default values, will be modified)
	self._cost = self._generativecost
	# gradients
        self._grads = T.grad(self._cost, self.params)


        # COMPILE SOME FUNCTIONS THAT MAY BE USEFUL LATER 
        self.classes = theano.function([self.inputs], self.layer._class)
        self.reconsModality1 = theano.function([self.inputs], self._reconsModality1)
        self.reconsModality2 = theano.function([self.inputs], self._reconsModality2)
        self.reconsModality3 = theano.function([self.inputs], self._reconsModality3)
        self.cost = theano.function([self.inputs], self._cost)
        self.grads = theano.function([self.inputs], self._grads)
       
       
        c = T.matrix() # virtual classes
        p = T.matrix() # virtual parameters

	# generate reconstructions of each modality given class and params
        self.generateMod1 = theano.function([c,p],self.layer._modality1,givens=[(self.layer._corruptedClass,c),(self.layer._corruptedParam,p)])
        self.generateMod2 = theano.function([c,p],self.layer._modality2,givens=[(self.layer._corruptedClass,c),(self.layer._corruptedParam,p)])
        self.generateMod3 = theano.function([c,p],self.layer._modality3,givens=[(self.layer._corruptedClass,c),(self.layer._corruptedParam,p)])
        
	# generate params from each modality
        self.paramFrom1 = theano.function([self.layer._corruptedClass,self.layer._corruptedModality1], self.layer._paramFrom1)
        self.paramFrom2 = theano.function([self.layer._corruptedClass,self.layer._corruptedModality2], self.layer._paramFrom2)
        self.paramFrom3 = theano.function([self.layer._corruptedClass,self.layer._corruptedModality3], self.layer._paramFrom3)
        self.paramFrom12 = theano.function([self.layer._corruptedClass,self.layer._corruptedModality1, self.layer._corruptedModality2], self.layer._paramFrom12)
        self.paramFrom13 = theano.function([self.layer._corruptedClass,self.layer._corruptedModality1, self.layer._corruptedModality3], self.layer._paramFrom13)
        self.paramFrom23 = theano.function([self.layer._corruptedClass,self.layer._corruptedModality2, self.layer._corruptedModality3], self.layer._paramFrom23)
   
   	# generate classes given some modalities
        self.classFrom1 = theano.function([self.layer._corruptedModality1], self.layer._classFrom1)
        self.classFrom2 = theano.function([self.layer._corruptedModality2], self.layer._classFrom2)
        self.classFrom3 = theano.function([self.layer._corruptedModality3], self.layer._classFrom3)
        self.classFrom12 = theano.function([self.layer._corruptedModality1, self.layer._corruptedModality2], self.layer._classFrom12)
        self.classFrom13 = theano.function([self.layer._corruptedModality1, self.layer._corruptedModality3], self.layer._classFrom13)
        self.classFrom23 = theano.function([self.layer._corruptedModality2, self.layer._corruptedModality3], self.layer._classFrom23)
       
        def get_cudandarray_value(x):
            if type(x)==theano.sandbox.cuda.CudaNdarray:
                return numpy.array(x.__array__()).flatten()
            else:
                return x.flatten()
        self.grad = lambda x: numpy.concatenate([get_cudandarray_value(g) for g in self.grads(x)])

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])



