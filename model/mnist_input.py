""" Input for mnist model"""

from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import os
import re
import numpy as np
import tensorflow as tf
import pandas as pd

IMAGE_SIZE = 28
NUM_CLASSES = 10
NUM_EXAMPLE_PER_EPOCH_FOR_TRAIN = 500
NUM_EXAMPLE_PER_EPOCH_FOR_EVAL = 100

def read_mnist(filename):
	train = pd.read_csv(filename)
	labels = train['label']
	images = train.drop(['label'],axis=1)
	images = images/255.
	return images,labels	

def _generate_image_and_label_batch(images,labels,min_queue_examples,
									batch_size,shuffle):
	""" Take a single image as queue and output a batch of images
	
	"""
	num_preprocess_threads = 16
	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
			[images,labels],
			batch_size = batch_size,
			num_threads=num_preprocess_threads,
			capacity = min_queue_examples+3*batch_size,
			min_after_dequeue=min_queue_examples,
			enqueue_many = True)
	else:
		images,label_batch = tf.train.batch(
			[images,labels],
			batch_size = batch_size,
			num_threads = nup_preprocess_threads,
			capacity = min_queue_examples+3*batch_size,
			enqueue_many = True)
	tf.summary.image('images',images)
	return images,label_batch

def distorted_inputs(data_dir,batch_size):
	with tf.name_scope("data_augmentation"):
		filename = os.path.join(data_dir,'train.csv')
		images,labels = read_mnist(filename)
		height = IMAGE_SIZE
		width = IMAGE_SIZE
		
		images = np.array(images).reshape([-1,height,width,1])	
		images = tf.cast(images,tf.float32)
		min_fraction_of_examples_in_queue = 0.4
		min_queue_samples = int(NUM_EXAMPLE_PER_EPOCH_FOR_TRAIN*
								min_fraction_of_examples_in_queue)
	return _generate_image_and_label_batch(images,labels,
											min_queue_samples,batch_size,
											shuffle=True)
def inputs(eval_data,data_dir,batch_size):
	
	if not eval_data:
		filenames = os.path.join(path,'train.csv')
		num_example_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		filename = os.path.join(path,'test.csv')
		num_example_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
	with tf.name_scope("input"):
		images,labels = read_mnist(filename)
		height = IMAGE_SIZE
		width = IMAGE_SIZE
		
		min_fraction_of_examples_in_queue = 0.4
		min_queue_samples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*
								min_fraction_of_examples_in_queue)
	return _generate_image_and_label_batch(float_image,label,
											min_queue_examples,batch_size,
											shuffle=False)

