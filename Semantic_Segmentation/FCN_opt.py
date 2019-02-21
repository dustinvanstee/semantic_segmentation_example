import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from PIL import Image

import tensorflow as tf
import pdb
import time
import helper_opt

# Notes
# /gpfs/b8s013/cssc/Semantic_Segmentation/Data/Road/data_road/training/image_2
# /gpfs/b8s013/cssc/Semantic_Segmentation
# source source activate nico-tf-gpu

# Tune these parameters 
NUMBER_OF_CLASSES = 2
IMAGE_SHAPE = (160, 576)
EPOCHS = 40
BATCH_SIZE = 16
DROPOUT = 0.75


# Specify these directory paths
data_dir = './Data/Road/'
runs_dir = './Runs'
training_dir ='./Data/Road/data_road/training'
vgg_path = './Data/vgg'

correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUMBER_OF_CLASSES])

learning_rate = tf.placeholder(tf.float32)

def load_vgg(sess, vgg_path):
	# load the model and weights    
	
	model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
	# Get Tensors to be returned from graph
	graph = tf.get_default_graph()
	image_input = graph.get_tensor_by_name('image_input:0')
	keep_prob = graph.get_tensor_by_name('keep_prob:0')
	layer3 = graph.get_tensor_by_name('layer3_out:0')
	layer4 = graph.get_tensor_by_name('layer4_out:0')
	layer7 = graph.get_tensor_by_name('layer7_out:0')
	return image_input, keep_prob, layer3, layer4, layer7

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
	layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

	fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, name="fcn8")
	
	fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1], kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")

	fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

	fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
	kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

	fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

	fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes, kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

	return fcn11



def optimize(nn_last_layer, correct_label, learning_rate, num_classes):

	logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
	correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])

	loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

	return logits, train_op, loss_op

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
	keep_prob_value = 0.5

	learning_rate_value = 0.001

	# preprocess all the images into ram ....

	for epoch in range(epochs):

		print("Timing epoch {}".format(epoch))
		epoch_t0 = time.time()
		total_loss = 0
		for X_batch, gt_batch in get_batches_fn(batch_size):

			loss, _ = sess.run([cross_entropy_loss, train_op],
			feed_dict={input_image: X_batch, correct_label: gt_batch,
			keep_prob: keep_prob_value, learning_rate:learning_rate_value})
			
			total_loss += loss;
		epoch_t1 = time.time()

		print("EPOCH {} ... {}".format(epoch + 1, epoch_t1 - epoch_t0))
		print("Loss = {:.3f}".format(total_loss))
		print()

def run():

#	helper.maybe_download_pretrained_vgg(vgg_path)
	start_time = time.time()
	#pdb.set_trace()

	t0 = time.time()
	image_target = helper_opt.read_images_opt(training_dir, IMAGE_SHAPE)
	t1 = time.time()
	print("####################")
	print("Read time = {}".format(t1-t0))
	print("####################")
	get_batches_fn = helper_opt.gen_batch_function_opt(image_target)


	config = tf.ConfigProto(device_count = {'GPU': 1})
	
#	config = tf.ConfigProto()
#	config.gpu_options.allow_growth=True

#	config.gpu_options.per_process_gpu_memory_fraction = 0.4

	with tf.Session(config=config) as session:		

#		---------------------------MobaTextEditor---------------------------Cannot open clipboard.---------------

		image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, vgg_path)

		model_output = layers(layer3, layer4, layer7, NUMBER_OF_CLASSES)

		logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, NUMBER_OF_CLASSES)

		session.run(tf.global_variables_initializer())
		session.run(tf.local_variables_initializer())

		print("Model built successfully, starting training")

		train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn,
		train_op, cross_entropy_loss, image_input,
		correct_label, keep_prob, learning_rate)

		helper_opt.save_inference_samples(runs_dir, data_dir, session, IMAGE_SHAPE, logits, keep_prob, image_input)
#helper.save_inference_samples(runs_dir, data_dir, session, IMAGE_SHAPE, logits, keep_prob, image_input)
		print("All done!")
		
		print("--- %s seconds ---" % (time.time() - start_time))

run()


