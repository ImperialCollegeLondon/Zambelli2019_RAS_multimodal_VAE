import pylab
import numpy
import numpy.random
import network_trimodal as network 
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import sys
import os
import shutil

class GraddescentMinibatch(object):
    """ Gradient descent trainer class.

    """

    def __init__(self, model, data, batchsize, learningrate, momentum=0.9, rng=None, verbose=True):
        self.model         = model
        self.data          = data
        self.learningrate  = learningrate
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value().shape[0] / batchsize
        self.momentum      = momentum 
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar() 
        self.incs = dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, 
                            dtype=theano.config.floatX), name='inc_'+p.name)) for p in self.model.params])
        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n
        self.set_learningrate(self.learningrate)

        self._generativecost = theano.function([self.index], self.model._generativecost,givens={self.model.inputs:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize]})
        self._reconscost = theano.function([self.index], self.model._reconscost,givens={self.model.inputs:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize]})


    def set_learningrate(self, learningrate):
    	# for each params, set up the update formula
        self.incs = dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, dtype=theano.config.floatX), name='inc_'+p.name)) for p in self.model.params])
        self.inc_updates = {}
        self.updates = {}
        self.learningrate = learningrate
        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * self.model.layer.learningrate_modifiers[_param.name] * _grad 
            self.updates[_param] = _param + self.incs[_param]


	# this function computes the update matrices using momentum
        self._updateincs = theano.function([self.index], self.model._cost, 
                                     updates = self.inc_updates,
                givens = {self.model.inputs:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize]})
	# this function update parameters with their updates computed by previous function
	self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)
   
   
    # performs one epoch
    def step(self, f):
        cost = 0.0
        generativecost=  0.0
        reconscost = 0.0
        stepcount = 0.0
        for batch_index in self.rng.permutation(self.numbatches-1):
            stepcount += 1.0
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
            generativecost = (1.0-1.0/stepcount)*generativecost + (1.0/stepcount)*self._generativecost(batch_index)
            reconscost = (1.0-1.0/stepcount)*reconscost + (1.0/stepcount)*self._reconscost(batch_index)
            self._trainmodel(0)
        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)
            f.write(str(cost)+" "+str(reconscost)+" "+str(generativecost)+"\r\n")
            f.flush()


print '... loading data'
train_features_mod1 = numpy.loadtxt('./pic_full.txt').astype(theano.config.floatX)
train_features_mod2 = numpy.loadtxt('./traj_full.txt').astype(theano.config.floatX)
train_features_mod3 = numpy.loadtxt('./speech_alain.txt').astype(theano.config.floatX)

#SHUFFLE TRAINING DATA TO MAKE SURE ITS NOT SORTED:
R = numpy.random.permutation(train_features_mod1.shape[0])
train_features_mod1 = train_features_mod1[R, :]
train_features_mod2 = train_features_mod2[R, :]
train_features_mod3 = train_features_mod3[R, :]


train_features_numpy = numpy.concatenate((train_features_mod1, train_features_mod2, train_features_mod3), 1)
train_features = theano.shared(train_features_numpy)
print '... done'

print '... loading test data'
test_features_mod1 = numpy.loadtxt('./pic_full.txt').astype(theano.config.floatX)
test_features_mod2 = numpy.loadtxt('./traj_full.txt').astype(theano.config.floatX)
test_features_mod3 = numpy.loadtxt('./speech_alain.txt').astype(theano.config.floatX)

test_features_numpy = numpy.concatenate((test_features_mod1, test_features_mod2, test_features_mod3), 1)
test_features = theano.shared(test_features_numpy)
print '... done'


# retrieve params from command line
numModel1 = int(sys.argv[1]) 
numFactor1 = int(sys.argv[2])
numModel2 = int(sys.argv[3]) 
numFactor2 = int(sys.argv[4])
numModel3 = int(sys.argv[5]) 
numFactor3 = int(sys.argv[6])
numClass = int(sys.argv[7])
numParam = int(sys.argv[8])
softmaxnoise = float(sys.argv[9])
corruption_level = 0.3
batchsize = 10
numModality1 = train_features_mod1.shape[1]
numModality2 = train_features_mod2.shape[1]
numModality3 = train_features_mod3.shape[1]
numbatches = train_features.get_value().shape[0] / batchsize
learningrate = 0.001


# log directory
directory='exp'+str(numModel1)+'_'+str(numFactor1)+'_'+str(numModel2)+'_'+str(numFactor2)+'_'+str(numModel3)+'_'+str(numFactor3)+'_'+str(numClass)+'_'+str(numParam)+str(corruption_level)+'_'+str(batchsize)+'_'+str(softmaxnoise)+'_'+str(learningrate)

try:
        os.makedirs("logs/"+directory)
except OSError:
        print directory+' already exists, do you want to continue ?'
        sys.stdin.read(1) #this is to wait for CTRL+C

# copy the current code source with their logs
shutil.copy('network_trimodal.py','logs/'+directory)
shutil.copy('network_trimodal_use.py','logs/'+directory)
# file to log the cost at each epoch
f=open("logs/"+directory+"/log.txt","w")

# INSTANTIATE MODEL
print '... instantiating model'
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)
model = network.Network(numModality1=numModality1, numModel1=numModel1, numFactor1=numFactor1, numModality2=numModality2, numModel2=numModel2, numFactor2=numFactor2, numModality3=numModality3, numModel3=numModel3, numFactor3=numFactor3, numClass=numClass, numParam=numParam,
                                                  corruption_level=corruption_level, 
                                                  softmaxnoise=softmaxnoise,
                                                  numpy_rng=numpy_rng, 
                                                  theano_rng=theano_rng)

print '... done'



# TRAIN MODEL
numepochs = 10001
trainer = GraddescentMinibatch(model, train_features, batchsize, learningrate)

for epoch in xrange(numepochs):
	# start by training only the autoencoders, for the 3000 first steps
        if (trainer.epochcount == 0):
                model.params=[model.layer.wModalityModel1,model.layer.wModalityModel2,model.layer.wModalityModel3,model.layer.bModality1,model.layer.bModality2,model.layer.bModality3,model.layer.bModel1,model.layer.bModel2,model.layer.bModel3]
                model._cost = model._reconscost
                model._grads = T.grad(model._cost, model.params)
                model.cost = theano.function([model.inputs], model._cost)
                model.grads = theano.function([model.inputs], model._grads)
                # the call to set_learningrate updates all the gradient descent setup
		trainer.set_learningrate(learningrate)
	# then train the gated network for the next 3000 steps
        elif (trainer.epochcount == 3000):
                model.params=[model.layer.wModelClass1,model.layer.wModelClass2, model.layer.wModelClass3, model.layer.wModelFactor1, model.layer.wModelFactor2, model.layer.wModelFactor3, model.layer.wClassFactor1,model.layer.wClassFactor2, model.layer.wClassFactor3, model.layer.wFactorParam1, model.layer.wFactorParam2, model.layer.wFactorParam3, model.layer.bClass, model.layer.bParam]
                model._cost = model._generativecost
                model._grads = T.grad(model._cost, model.params)
                model.cost = theano.function([model.inputs], model._cost)
                model.grads = theano.function([model.inputs], model._grads)
                trainer.set_learningrate(learningrate)
        # finish by finetuning the whole network...
	elif (trainer.epochcount == 6000):
                model.params=[model.layer.wModalityModel1,model.layer.wModalityModel2,model.layer.wModalityModel3,model.layer.bModality1,model.layer.bModality2,model.layer.bModality3, model.layer.bModel1,model.layer.bModel2, model.layer.bModel3, model.layer.wModelClass1,model.layer.wModelClass2, model.layer.wModelClass3, model.layer.wModelFactor1, model.layer.wModelFactor2, model.layer.wModelFactor3, model.layer.wClassFactor1,model.layer.wClassFactor2, model.layer.wClassFactor3, model.layer.wFactorParam1, model.layer.wFactorParam2, model.layer.wFactorParam3, model.layer.bClass, model.layer.bParam]
                model._cost = model._generativecost
                model._grads = T.grad(model._cost, model.params)
                model.cost = theano.function([model.inputs], model._cost)
                model.grads = theano.function([model.inputs], model._grads)
                trainer.set_learningrate(learningrate)
        # learning rate annealing
	elif ((trainer.epochcount % 100) == 0):
                trainer.set_learningrate(trainer.learningrate*0.99)
                print 'New learning rate : '+str(trainer.learningrate)
        # periodic log of several values
	if ((trainer.epochcount % 1000) == 0):
                # global classification
		out = model.classes(test_features.get_value())
                numpy.savetxt("logs/"+directory+"/class_iter"+str(trainer.epochcount)+".txt",out)
                
		# use only mod1 to classify, and reconstruct the other modalities from it
                c=model.classFrom1(test_features_mod1)
                numpy.savetxt("logs/"+directory+"/classFrom1_iter"+str(trainer.epochcount)+".txt",c)
                p=model.paramFrom1(c,test_features_mod1)
                out = model.generateMod2(c,p)
                numpy.savetxt("logs/"+directory+"/2From1_iter"+str(trainer.epochcount)+".txt",out)
                out = model.generateMod3(c,p)
                numpy.savetxt("logs/"+directory+"/3From1_iter"+str(trainer.epochcount)+".txt",out)
                
                # idem with mod2
                c=model.classFrom2(test_features_mod2)
                numpy.savetxt("logs/"+directory+"/classFrom2_iter"+str(trainer.epochcount)+".txt",c)
                p=model.paramFrom2(c,test_features_mod2)
                out = model.generateMod1(c,p)
                numpy.savetxt("logs/"+directory+"/1From2_iter"+str(trainer.epochcount)+".txt",out)
                out = model.generateMod3(c,p)
                numpy.savetxt("logs/"+directory+"/3From2_iter"+str(trainer.epochcount)+".txt",out)
        
                # idem with mod3
                c=model.classFrom3(test_features_mod3)
                numpy.savetxt("logs/"+directory+"/classFrom3_iter"+str(trainer.epochcount)+".txt",c)
                p=model.paramFrom3(c,test_features_mod3)
                out = model.generateMod1(c,p)
                numpy.savetxt("logs/"+directory+"/1From3_iter"+str(trainer.epochcount)+".txt",out)
                out = model.generateMod2(c,p)
                numpy.savetxt("logs/"+directory+"/2From3_iter"+str(trainer.epochcount)+".txt",out)
        
		# idem with mod1+2
                c=model.classFrom12(test_features_mod1, test_features_mod2)
                numpy.savetxt("logs/"+directory+"/classFrom12_iter"+str(trainer.epochcount)+".txt",c)
                p=model.paramFrom12(c,test_features_mod1, test_features_mod2)
                out = model.generateMod3(c,p)
                numpy.savetxt("logs/"+directory+"/3From12_iter"+str(trainer.epochcount)+".txt",out)
        
		# idem with mod1+3
                c=model.classFrom13(test_features_mod1, test_features_mod3)
                numpy.savetxt("logs/"+directory+"/classFrom13_iter"+str(trainer.epochcount)+".txt",c)
                p=model.paramFrom13(c,test_features_mod1, test_features_mod3)
                out = model.generateMod2(c,p)
                numpy.savetxt("logs/"+directory+"/2From13_iter"+str(trainer.epochcount)+".txt",out)
        
		# idem with mod2+3
                c=model.classFrom23(test_features_mod2, test_features_mod3)
                numpy.savetxt("logs/"+directory+"/classFrom23_iter"+str(trainer.epochcount)+".txt",c)
                p=model.paramFrom23(c,test_features_mod2, test_features_mod3)
                out = model.generateMod1(c,p)
                numpy.savetxt("logs/"+directory+"/1From23_iter"+str(trainer.epochcount)+".txt",out)
        


        trainer.step(f)


f.close()
