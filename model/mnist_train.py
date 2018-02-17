"""Training file"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import pandas as pd
import tensorflow as tf
import time
import mnist
from datetime import datetime
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir','/tmp/mnist_train',
							"""Directory to write event logs and ckt""")
tf.app.flags.DEFINE_integer('max_steps',1000000,
							"""Number of batches to run""")
tf.app.flags.DEFINE_boolean('log_device_placement',False,
							"""whether to log device placement""")
tf.app.flags.DEFINE_integer("log_frequency",10,
							"""How often to log results""")
	
def train():
	
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()
		
		with tf.device('/cpu:0'):
			images,labels = mnist.distorted_inputs()
		print(global_step)
		
		#with tf.device('/gpu:0'):
		logits = mnist.inference(images)
		#with tf.device('/gpu:0'):
		loss = mnist.loss(logits,labels)
		#with tf.device('/gpu:0'):	
		train_op = mnist.train(loss,global_step)
			
		class _LoggerHook(tf.train.SessionRunHook):
			def begin(self):
				self._step=-1
				self._start_time = time.time()
			def before_run(self,run_context):
				self._step+=1
				return tf.train.SessionRunArgs(loss)
			def after_run(self,run_context,run_values):
				if self._step % FLAGS.log_frequency == 0:
					current_time = time.time()
					duration = current_time-self._start_time
					self._start_time = current_time

					loss_value = run_values.results
					examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
					sec_per_batch = float(duration / FLAGS.log_frequency)

					format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
									'sec/batch)')
					print (format_str % (datetime.now(), self._step, loss_value,
							examples_per_sec, sec_per_batch))
		
		with tf.train.MonitoredTrainingSession(
			checkpoint_dir = FLAGS.train_dir,
			hooks =[tf.train.StopAtStepHook(last_step = FLAGS.max_steps),
					tf.train.NanTensorHook(loss),
					_LoggerHook()],
			config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
			while not mon_sess.should_stop():
				mon_sess.run(train_op)

def main(argv=None):
	train()


if __name__ == '__main__':
	tf.app.run()











