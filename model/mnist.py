


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
 
import numpy as np
import os 
import re
import sys
import tensorflow as tf

#from six.moves import urllib
import mnist_input

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size',128,
				"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir','../input/',"""Path to MNIST dataset.""")
tf.app.flags.DEFINE_bool('use_fp16',False,"""Train the model using fp16.""")
IMAGE_SIZE = mnist_input.IMAGE_SIZE
NUM_CLASSES = mnist_input.NUM_CLASSES
NUM_EXAMPLE_PER_EPOCH_FOR_TRAIN = mnist_input.NUM_EXAMPLE_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLE_PER_EPOCH_FOR_EVAL = mnist_input.NUM_EXAMPLE_PER_EPOCH_FOR_EVAL

# Constants describing the training process
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 305.0 
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

def _activation_summary(x):
	"""Helper to create summaries for the activation
	Create a summary that provides a histogram of activations.
	Create a summary that provides a sparsity of activations.
	
	Args:
		x: Tensor
	Returns:
		nothing
	"""
	
	tf.summary.histogram(x.op.name+'/activations',x)
	tf.summary.scalar(x.op.name+'/sparsity',tf.nn.zero_fraction(x))

def _variable_on_cpu(name,shape,initializer):
	"""Helper to create a Variable stored on CPU memory.	
	
	ARGS:
		name: name of the variable.
		shape: lists of int.
		initializer: initializer for Variable.
	Returns:
		Variable Tensor.
	"""
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name,shape,
								initializer = initializer,dtype = dtype)
	return var

def _variable_with_weight_decay(name,shape,stddev,wd):
	""" Helper to create an initialized variable with weight decay.
	"""
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _variable_on_cpu(
		name,
		shape,
		tf.truncated_normal_initializer(stddev=stddev,dtype=dtype))
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
		tf.add_to_collection('losses',weight_decay)
	return var

def distorted_inputs():
	"""Construct distorted input for MNIST
	Returns:
		images: Images.
	"""
	if not FLAGS.data_dir:
		raise ValueError('please supply a data_dir')    
	images,labels = mnist_input.distorted_inputs(data_dir=FLAGS.data_dir,
												batch_size=FLAGS.batch_size)
	if FLAGS.use_fp16:
		images = tf.cast(images,tf.float16)
		labels = tf.cast(labels,tf.float16)
	return images,labels	

def inputs(eval_data):
	images,labels = mnist_input.inputs(data_dir=FLAGS.data_dir,
										batch_size = FLAGS.batch_size)
	if FLAGS.use_fp16:
		images = tf.cast(images,tf.float16)
		labels = tf.cast(images,tf.float16)
def inference(images):
	
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weigths',
											shape=[5,5,1,64],
											stddev = 5e-2,	
											wd=None)
		conv = tf.nn.conv2d(images,kernel,[1,1,1,1],padding='SAME')
		biases = _variable_on_cpu('biases',[64],tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv,biases)
		conv1 = tf.nn.relu(pre_activation,name=scope.name)
		_activation_summary(conv1)
	
	pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],
							padding='SAME',name='pool1')
	norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,
						beta=0.75,name='norm1')
	
	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[5,5,64,64],
											 stddev=5e-2,
											 wd=None)
		conv = tf.nn.conv2d(norm1,kernel,[1,1,1,1],padding='SAME')
		biases = _variable_on_cpu('biases',[64],
									tf.constant_initializer(0.1))
		pre_activation = tf.nn.bias_add(conv,biases)
		conv2 = tf.nn.relu(pre_activation,name=scope.name)
		_activation_summary(conv2)
	norm2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm2')
	pool2 = tf.nn.max_pool(norm2,[1,3,3,1],strides=[1,2,2,1],
							padding='SAME',name='pool2')
	
	with tf.variable_scope('local3') as scope:
		reshape = tf.reshape(pool2,[FLAGS.batch_size,-1]) 
		dim = reshape.get_shape()[1].value
		weights =  _variable_with_weight_decay('weigts',shape=[dim,384],
												stddev=0.04,wd=0.004)
		biases = _variable_on_cpu('biases',[384],tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)
		_activation_summary(local3)

	with tf.variable_scope('local4') as scope:
		weights = _variable_with_weight_decay('weights',shape = [384,192],
												stddev = 0.04,wd=0.004)
		biases = _variable_on_cpu('biases',[192],tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3,weights)+biases,name=scope.name)
		_activation_summary(local4)
	
	with tf.variable_scope('softmax_linear') as scope:
		weights = _variable_with_weight_decay('weights',[192,NUM_CLASSES],
											  stddev = 1/192.0,wd=None)
		biases = _variable_on_cpu('biases',[NUM_CLASSES],
									tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(local4,weights),biases,
									name='scope.name')
		_activation_summary(softmax_linear)    
	
	return softmax_linear

def loss(logits,labels):
	labels = tf.cast(labels,tf.int64)
	print(labels.shape)
	print(logits.shape)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
	tf.add_to_collection('losses',cross_entropy_mean)
	return tf.add_n(tf.get_collection('losses'),name='total_loss')

def _add_loss_summaries(total_loss):
	loss_average = tf.train.ExponentialMovingAverage(0.9,name='avg')
	losses = tf.get_collection('losses')
	loss_average_op = loss_average.apply(losses+[total_loss])
	
	for l in losses+[total_loss]:
		tf.summary.scalar(l.op.name+'(raw)',l)
		tf.summary.scalar(l.op.name,loss_average.average(l))
	
	return loss_average_op 
	
def train(total_loss,global_step):
	num_batches_per_epoch = NUM_EXAMPLE_PER_EPOCH_FOR_TRAIN/FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY)
	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
									global_step,
									decay_steps,
									LEARNING_RATE_DECAY_FACTOR,
									staircase=True)
	tf.summary.scalar('learrning_rate',lr)
	loss_average_op = _add_loss_summaries(total_loss)

	with tf.control_dependencies([loss_average_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(total_loss)		
	
	apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)
	
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name,var)
	
	for grad, var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name+'/gradients',grad)
	
	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY,global_step)
	variable_averages_op = variable_averages.apply(tf.trainable_variables())
	with tf.control_dependencies([apply_gradient_op,variable_averages_op]):
		train_op = tf.no_op(name='train')
	
	return train_op



	
