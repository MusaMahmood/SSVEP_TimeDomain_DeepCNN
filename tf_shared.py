import glob
import numpy as np
import pandas as pd
import winsound as ws
import tensorflow as tf
import time
from scipy.io import loadmat, savemat
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


def tf_initialize():
    # Merges all summaries collected in the default graph.
    tf.summary.merge_all()
    saver = tf.train.Saver()  # Initialize tf Saver for model export
    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return saver, init_op, config


def train(x, y, keep_prob, accuracy, train_step, x_train, y_train, x_test, y_test, keep_prob_feed=0.5,
          number_steps=100, batch_size=64, test_batch_size=100, train_check=10, test_check=20):
    val_step = 0
    total_val_steps = number_steps // 20
    val_accuracy_array = np.zeros([total_val_steps, 2], dtype=np.float32)
    for i in range(0, number_steps):
        offset = (i * batch_size) % (x_train.shape[0] - batch_size)
        batch_x_train = x_train[offset:(offset + batch_size)]
        batch_y_train = y_train[offset:(offset + batch_size)]
        if i % train_check == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_x_train, y: batch_y_train, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        if i % test_check == 0:
            offset = (val_step * test_batch_size) % (x_test.shape[0] - test_batch_size)
            batch_x_val = x_test[offset:(offset + test_batch_size), :, :]
            batch_y_val = y_test[offset:(offset + test_batch_size), :]
            val_accuracy = accuracy.eval(feed_dict={x: batch_x_val, y: batch_y_val, keep_prob: 1.0})
            print("Validation step %d, validation accuracy %g" % (val_step, val_accuracy))
            val_accuracy_array[val_step, 0] = (1 + val_step) * 20 * test_batch_size
            val_accuracy_array[val_step, 1] = val_accuracy
            val_step += 1

        train_step.run(feed_dict={x: batch_x_train, y: batch_y_train, keep_prob: keep_prob_feed})


def test(sess, x, y, accuracy, x_test, y_test, keep_prob, test_type='Holdout Validation'):
    test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.0})
    print("Testing Accuracy - ", test_type, ':', test_accuracy, "\n\n")
    return test_accuracy


def confusion_matrix_test(sess, x, y, keep_prob, prediction, input_shape, x_val_data, y_val_data, number_classes=5):
    y_val_tf = np.zeros([x_val_data.shape[0]], dtype=np.int32)
    predictions = np.zeros([x_val_data.shape[0]], dtype=np.int32)
    for i in range(0, x_val_data.shape[0]):
        predictions[i] = sess.run(prediction,
                                  feed_dict={x: x_val_data[i].reshape(input_shape),
                                             y: y_val_data[i].reshape([1, number_classes]), keep_prob: 1.0})
        for c in range(0, number_classes):
            if y_val_data[i][c]:
                y_val_tf[i] = c

    tf_confusion_matrix = tf.confusion_matrix(labels=y_val_tf, predictions=predictions, num_classes=number_classes)
    confusion_matrix_nd = tf.Tensor.eval(tf_confusion_matrix, feed_dict=None, session=None)
    print(confusion_matrix_nd)
    return confusion_matrix_nd


def current_time_ms():
    return int(round(time.time() * 1000))


def placeholders(input_shape, number_classes, input_node_name, keep_prob_node_name, dtype=tf.float32):
    x = tf.placeholder(dtype, shape=[None, *input_shape], name=input_node_name)
    keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
    y = tf.placeholder(dtype, shape=[None, number_classes])
    return x, y, keep_prob


def reshape_input(x, input_shape):
    return tf.reshape(x, [-1, *input_shape, 1])


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


def conv_layer(input_, w_kernels, in_ch, num_kernels, strides, activation='relu', alpha=0.01):
    """
    Quick function for returning relevant data
    :param input_: Input Tensor
    :param w_kernels: kernel size [x, y] (e.g. [3, 3])
    :param in_ch: number of channels coming in from previous layer
    :param num_kernels: Number of kernel convolutions to output
    :param strides: conv stride [x, y], (e.g. [1, 2])
    :param activation: Activation function
    Options for activation are :
        'relu'
        'elu'
        'leakyrelu'
        'parametricrelu'
    :param alpha: If using parametric relu
    :return: Tensor representing the current layer.
    """
    weights, biases = var_weight_bias([*w_kernels, in_ch, num_kernels], [num_kernels])
    return conv(input_, weights, biases, stride=[1, *strides, 1], activation=activation, alpha=alpha)


def fully_connect_layer(x, w_shape, b_shape, do, keep_prob, activation='relu', alpha_fc=0.01):
    w_f, b_f = var_weight_bias(w_shape, b_shape)
    return fully_connect(x, w_f, b_f, keep_prob, do, activation, alpha_fc)


def output_layer(x, w_shape, b_shape, version='v1'):
    w_fco, b_fco = var_weight_bias(w_shape, b_shape)
    if version == 'v1':
        return connect(x, w_fco, b_fco)
    elif version == 'v2':
        return connect_v2(x, w_fco, b_fco)


def conv(x, w, b, stride=list([1, 1, 1, 1]), activation='relu', padding='SAME', alpha=0.01):
    x = tf.nn.conv2d(x, w, strides=stride, padding=padding)
    x = tf.nn.bias_add(x, b)
    if activation == 'relu':
        return tf.nn.relu(x)
    elif activation == 'elu':
        return tf.nn.elu(x)
    elif activation == 'leakyrelu':
        return tf.nn.leaky_relu(x, alpha=0.01)
    elif activation == 'parametricrelu':
        return tf.nn.leaky_relu(x, alpha=alpha)
    elif activation == 'crelu':
        return tf.nn.crelu(x)
    elif activation == 'selu':
        return tf.nn.selu(x)
    # TODO: Parametric ELU?


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


# For a relu activated FC
def fully_connect(x, w, b, keep_prob, do='dropout', activation='relu', alpha=0.01):
    if activation == 'elu':
        fc = tf.nn.elu(connect(x, w, b))
    elif activation == 'leakyrelu':
        fc = tf.nn.leaky_relu(connect(x, w, b), alpha=0.01)
    elif activation == 'parametricrelu':
        fc = tf.nn.leaky_relu(connect(x, w, b), alpha=alpha)
    else:
        fc = tf.nn.relu(connect(x, w, b))
    if do == 'dropout':
        return tf.nn.dropout(fc, keep_prob=keep_prob)


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


def train_optimize(learning_rate, cross_entropy):
    return tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


def train_loss(y, y_conv, learning_rate):
    # Compute Cross entropy:
    cross_entropy = loss_layer_v2(y, y_conv)
    # Optimizing using Adam Optimizer
    return train_optimize(learning_rate, cross_entropy)


def get_outputs(y, y_conv, output_node_name):
    outputs = tf.nn.softmax(y_conv, name=output_node_name)
    prediction_check, prediction = check_prediction(y, outputs)
    accuracy = get_accuracy(prediction_check)
    return outputs, prediction, accuracy


def check_prediction(y, outputs):
    prediction = tf.argmax(outputs, 1)
    correct_class = tf.argmax(y, 1)
    return tf.equal(prediction, correct_class), prediction


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


# # # FOR SAVING DATA:
def get_model_description(num_layers, activation, do, keep_prob, units_fc, fc_activation,
                          lr_coeff, lr_exp, num_filters, conv_alpha=0.01, fc_alpha=0.01):
    model_description = 'CNN-' + str(num_layers) + "-a." + activation
    if activation == 'parametricrelu':
        model_description += '.' + str(conv_alpha)
    if do == 'dropout':
        model_description += '-drop' + str(keep_prob)
    model_description += '-fc.' + str(units_fc) + '.' + fc_activation
    if fc_activation == 'parametricrelu':
        model_description += '.' + str(fc_alpha)
    model_description += '-lr.' + str(lr_coeff) + 'e-' + str(lr_exp) + '-k.' + str(num_filters[0:num_layers])
    return model_description


def get_model_dimensions(h, h_flat, h_fc, y_conv, number_layers):
    message = ''
    message += "Model Dimensions: " + '\n'
    for i in range(0, number_layers):
        message += "h_conv" + str(i + 1) + ": " + str(get_tensor_shape_tuple(h[i])) + '\n'
    message += "h_flat: " + str(get_tensor_shape_tuple(h_flat)) + '\n'
    message += "h_fc: " + str(get_tensor_shape_tuple(h_fc)) + '\n'
    message += "y_conv: " + str(get_tensor_shape_tuple(y_conv)) + '\n'
    return message


def get_filter_dimensions(w_x, w_y, s_x, s_y, alpha_conv, num_layers):
    message = "Filter Dimensions:" + '\n'
    for i in range(0, num_layers):
        message += "h_c" + str(i + 1) + "_filt: " + str([w_x[i], w_y[i]]) + \
                   " stride: " + str([s_x[i], s_y[i]]) + " calpha=" + str(alpha_conv[i]) + '\n'
    return message


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


def beep(freq=900, length_ms=1000):
    # A 900Hz Beep:
    ws.Beep(freq, length_ms)
