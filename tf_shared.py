import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import loadmat, savemat
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


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


def save_statistics(folder_name, val_acc, file_name='stats.mat'):
    savemat(folder_name + file_name, mdict={'training_rate': val_acc})


# Model Building Macros: #
def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution and max-pooling functions
def conv(x_, w_, b_, stride, padding='SAME'):
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


def flatten(x):
    # dimensions
    shape = np.asarray(x.get_shape().as_list())
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


def fully_connect_elu_dropout(x, w, b, keep_prob):
    return tf.nn.dropout(fully_connect(x, w, b, activation='elu'), keep_prob=keep_prob)


def fully_connect_relu_dropout(x, w, b, keep_prob):
    return tf.nn.dropout(fully_connect(x, w, b, activation='relu'), keep_prob=keep_prob)


def fully_connect_leakyrelu_dropout(x, w, b, keep_prob, alpha=0.01):
    return tf.nn.dropout(fully_connect(x, w, b, 'leakyrelu', alpha=alpha), keep_prob=keep_prob)


# For a relu activated FC
def fully_connect(x, w, b, activation='relu', alpha=0.01):
    if activation == 'relu':
        return tf.nn.relu(connect(x, w, b))
    elif activation == 'elu':
        return tf.nn.elu(connect(x, w, b))
    elif activation == 'leakyrelu':
        return tf.nn.leaky_relu(connect(x, w, b), alpha=alpha)


# Simple matmul + bias add. Used for output layer
def connect(x, w, b):
    return tf.add(tf.matmul(x, w), b)


def connect_v2(x, w, b):
    return tf.nn.bias_add(tf.matmul(x, w), b)


def loss_layer(y, y_conv):
    # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_conv))
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))


def get_accuracy(correct_prediction):
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train(learning_rate, cross_entropy):
    return tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


def check_prediction(y, outputs):
    return tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1))


def get_outputs(y, node_name):
    return tf.nn.softmax(y, node_name)

# ## MATLAB FUNCTIONS FOR SAVING DATA
# def get_activations(layer, input_val, shape, directory, file_name, sum_all=False):
#     os.makedirs(directory)
#     units = sess.run(layer, feed_dict={x: np.reshape(input_val, shape, order='F'), keep_prob: 1.0})
#     print("units.shape: ", units.shape)
#     # plot_nn_filter(units, directory + file_name, True)
#     new_shape = [units.shape[1], units.shape[2]]
#     feature_maps = units.shape[3]
#     filename_ = directory + file_name
#     if sum_all:
#         new_array = np.reshape(units.sum(axis=3), new_shape)
#         pd.DataFrame(new_array).to_csv(filename_ + '_weight_matrix' + '.csv', index=False, header=False)
#         major_axis = np.argmax(new_array.shape)
#         summed_array = new_array.sum(axis=major_axis)
#         pd.DataFrame(summed_array).to_csv(filename_ + '_sum_all' + '.csv', index=False, header=False)
#         print('All Values:')
#         return summed_array
#     else:
#         for i0 in range(feature_maps):
#             pd.DataFrame(units[:, :, :, i0].reshape(new_shape)).to_csv(
#                 filename_ + '_' + str(i0 + 1) + '.csv', index=False, header=False)
#         return units
