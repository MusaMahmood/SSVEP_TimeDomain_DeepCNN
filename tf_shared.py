import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from scipy.io import loadmat, savemat
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


def current_time_ms():
    return int(round(time.time() * 1000))


# Load Data:
def load_data(data_directory, image_shape, key_x, key_y):
    x_train_data = np.empty([0, *image_shape], np.float32)
    y_train_data = np.empty([0], np.float32)
    training_files = glob.glob(data_directory + "/*.mat")
    for f in training_files:
        x_array = loadmat(f).get(key_x)
        y_array = loadmat(f).get(key_y)
        y_array = y_array.reshape([np.amax(y_array.shape)])
        x_train_data = np.concatenate((x_train_data, x_array), axis=0)
        y_train_data = np.concatenate((y_train_data, y_array), axis=0)
    y_train_data = np.asarray(pd.get_dummies(y_train_data).values).astype(np.float32)
    # return data_array
    print("Loaded Data Shape: X:", x_train_data.shape, " Y: ", y_train_data.shape)
    return x_train_data, y_train_data


# Save graph/model:
def export_model(input_node_names, output_node_name_internal, export_dir, model_name):
    freeze_graph.freeze_graph(export_dir + model_name + '.pbtxt', None, False,
                              export_dir + model_name + '.ckpt', output_node_name_internal, "save/restore_all",
                              "save/Const:0", export_dir + '/frozen_' + model_name + '.pb', True, "")
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(export_dir + '/frozen_' + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name_internal], tf.float32.as_datatype_enum)
    with tf.gfile.FastGFile(export_dir + '/opt_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Graph Saved - Output Directories: ")
    print("1 - Standard Frozen Model:", export_dir + '/frozen_' + model_name + '.pb')
    print("2 - Android Optimized Model:", export_dir + '/opt_' + model_name + '.pb')


def save_statistics(folder_name, val_acc, details, info, elapsed_time, test_accuracy, file_name='stats.mat'):
    savemat(folder_name + file_name, mdict={'training_rate': val_acc, 'details': details, 'info': info,
                                            'elapsed_time': elapsed_time, 'test_accuracy': test_accuracy})


def var_weight_bias(w_shape, b_shape):
    w = tf.Variable(tf.truncated_normal(w_shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=b_shape))
    return w, b


# Model Building Macros: #
def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv(x_, w_, b_, stride=list([1, 1, 1, 1]), activation='relu', padding='SAME', alpha=0.01):
    """
        Options for activation are :
        'relu'
        'elu'
        'leakyrelu'
        'parametricrelu'
    """
    x_ = tf.nn.conv2d(x_, w_, strides=stride, padding=padding)
    x_ = tf.nn.bias_add(x_, b_)
    if activation == 'relu':
        return tf.nn.relu(x_)
    elif activation == 'elu':
        return tf.nn.elu(x_)
    elif activation == 'leakyrelu':
        return tf.nn.leaky_relu(x_, alpha=0.01)
    elif activation == 'parametricrelu':
        return tf.nn.leaky_relu(x_, alpha=alpha)


# Convolution and max-pooling functions
def relu_conv(x_, w_, b_, stride, padding='SAME'):
    # INPUT: [batch, in_height, in_width, in_channels]
    x_ = tf.nn.conv2d(x_, w_, strides=stride, padding=padding)
    x_ = tf.nn.bias_add(x_, b_)
    return tf.nn.relu(x_)


def leaky_conv(x_, w_, b_, stride, alpha, padding='SAME'):
    x_ = tf.nn.conv2d(x_, w_, strides=stride, padding=padding)
    x_ = tf.nn.bias_add(x_, b_)
    return tf.nn.leaky_relu(x_, alpha=alpha)


def elu_conv(x_, w_, b_, stride, padding='SAME'):
    x_ = tf.nn.conv2d(x_, w_, strides=stride, padding=padding)
    x_ = tf.nn.bias_add(x_, b_)
    return tf.nn.elu(x_)


def get_tensor_shape_tuple(x_):
    shape_as_list = x_.get_shape().as_list()
    shape_as_list = list(filter(None.__ne__, shape_as_list))
    return tuple([1, *shape_as_list])


def get_tensor_shape(x_):
    shape_as_list = x_.get_shape().as_list()
    # filter out  'None' type:
    shape_as_list = list(filter(None.__ne__, shape_as_list))
    return np.asarray(shape_as_list)


def flatten(x):
    # dimensions
    # shape = np.asarray(x.get_shape().as_list())
    shape = get_tensor_shape(x)
    dense_shape = 1
    # check if dimension is valid before multiplying out
    for i in range(0, shape.shape[0]):
        if shape[i] is not None:
            dense_shape = dense_shape * shape[i]
    return tf.reshape(x, [-1, dense_shape]), dense_shape


# Zero Padded Max Pooling
def max_pool(x_, ksize, stride, padding='SAME'):
    return tf.nn.max_pool(x_, ksize=ksize, strides=stride, padding=padding)


def max_pool_valid(x_, ksize, stride):
    return tf.nn.max_pool(x_, ksize=ksize, strides=stride, padding='VALID')


def fully_connect_with_dropout(x, w, b, keep_prob, activation='relu', alpha=0.01):
    return tf.nn.dropout(fully_connect(x, w, b, activation=activation, alpha=alpha), keep_prob=keep_prob)


# For a relu activated FC
def fully_connect(x, w, b, activation='relu', alpha=0.01):
    if activation == 'relu':
        return tf.nn.relu(connect(x, w, b))
    elif activation == 'elu':
        return tf.nn.elu(connect(x, w, b))
    elif activation == 'leakyrelu':
        return tf.nn.leaky_relu(connect(x, w, b), alpha=0.01)
    elif activation == 'parametricrelu':
        return tf.nn.leaky_relu(connect(x, w, b), alpha=alpha)


# Simple matmul + bias add. Used for output layer
def connect(x, w, b):
    return tf.add(tf.matmul(x, w), b)


def connect_v2(x, w, b):
    return tf.nn.bias_add(tf.matmul(x, w), b)


def loss_layer(y, y_conv):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))


def loss_layer_v2(y, y_conv):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_conv))


def get_accuracy(correct_prediction):
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train(learning_rate, cross_entropy):
    return tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


def check_prediction(y, outputs):
    prediction = tf.argmax(outputs, 1)
    correct_class = tf.argmax(y, 1)
    return tf.equal(prediction, correct_class), prediction


def get_outputs(y, node_name):
    return tf.nn.softmax(y, node_name)


# ## FOR SAVING DATA:
def get_activations_mat(x, keep_prob, sess, layer, input_sample, input_shape):
    units = sess.run(layer, feed_dict={x: np.reshape(input_sample, input_shape, order='F'), keep_prob: 1.0})
    return units


# TODO: Create custom function to take any number of layers.
def get_all_activations_4layer(sess, x, keep_prob, input_shape, training_data, folder_name, h_conv1, h_conv2, h_conv3,
                               h_conv4,
                               h_conv_flat, h_fc1_drop, y_conv):
    h_conv1_shape = get_tensor_shape(h_conv1)
    h_conv2_shape = get_tensor_shape(h_conv2)
    h_conv3_shape = get_tensor_shape(h_conv3)
    h_conv4_shape = get_tensor_shape(h_conv4)
    h_conv_flat_shape = get_tensor_shape(h_conv_flat)
    h_fc1_drop_shape = get_tensor_shape(h_fc1_drop)
    y_conv_shape = get_tensor_shape(y_conv)
    # Create empty arrays
    w_hconv1 = np.empty([0, *h_conv1_shape], np.float32)
    w_hconv2 = np.empty([0, *h_conv2_shape], np.float32)
    w_hconv3 = np.empty([0, *h_conv3_shape], np.float32)
    w_hconv4 = np.empty([0, *h_conv4_shape], np.float32)
    w_flat = np.empty([0, *h_conv_flat_shape], np.float32)
    w_hfc1_do = np.empty([0, *h_fc1_drop_shape], np.float32)
    w_y_out = np.empty([0, *y_conv_shape], np.float32)
    print('Getting all Activations: please wait... ')
    for it in range(0, training_data.shape[0]):
        if it % 100 == 0:
            print('Saved Sample #', it)
        sample = training_data[it]
        w_hconv1 = np.concatenate((w_hconv1, get_activations_mat(x, keep_prob, sess, h_conv1, sample, input_shape)),
                                  axis=0)
        w_hconv2 = np.concatenate((w_hconv2, get_activations_mat(x, keep_prob, sess, h_conv2, sample, input_shape)),
                                  axis=0)
        w_hconv3 = np.concatenate((w_hconv3, get_activations_mat(x, keep_prob, sess, h_conv3, sample, input_shape)),
                                  axis=0)
        w_hconv4 = np.concatenate((w_hconv4, get_activations_mat(x, keep_prob, sess, h_conv4, sample, input_shape)),
                                  axis=0)
        w_flat = np.concatenate((w_flat, get_activations_mat(x, keep_prob, sess, h_conv_flat, sample, input_shape)),
                                axis=0)
        w_hfc1_do = np.concatenate((w_hfc1_do,
                                    get_activations_mat(x, keep_prob, sess, h_fc1_drop, sample, input_shape)), axis=0)
        w_y_out = np.concatenate((w_y_out,
                                  get_activations_mat(x, keep_prob, sess, y_conv, sample, input_shape)), axis=0)
        # Save all activations:
    fn_out = folder_name + 'all_activations.mat'
    savemat(fn_out, mdict={'input_sample': training_data, 'h_conv1': w_hconv1, 'h_conv2': w_hconv2,
                           'h_flat': w_flat, 'h_fc1': w_hfc1_do,
                           'y_out': w_y_out})
