# MUSA MAHMOOD; DEOGRATIAS MZURIKWAO - Copyright 2018
# Python 3.6.1
# TF 1.5.0
# SEE PAPER ON DEEP CNN WITH NO MAX POOLING

# IMPORTS:
import tensorflow as tf
import numpy as np
import os as os
import datetime
import time
import winsound as ws
import tf_shared as tfs

from scipy.io import savemat
from sklearn.model_selection import train_test_split

MODEL_TYPE = '4-cnn-elu'

# CONSTANTS:
TIMESTAMP_START = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H.%M.%S')
wlens = [128, 192, 256, 384, 512]
w_sel = 2  # 0 -> 4
win_len = wlens[w_sel]
electrodes = ''
descriptor = 'time_domain_hpf_new'  # FIXED + TRUNCATED
subject_number = 99
subject_folder = '/S' + str(subject_number)
folder_name = ''

TOTAL_DATA_CHANNELS = 2
TRAINING_FOLDER_PATH = r'' + descriptor + subject_folder + '/' + 'w' + str(win_len)
TEST_FOLDER_PATH = TRAINING_FOLDER_PATH + '/v'
EXPORT_DIRECTORY = 'model_exports/'
MODEL_NAME = 'ssvep_net_2ch' + '_S' + str(subject_number) + '_' + descriptor + '_wlen' + str(win_len)
CHECKPOINT_FILE = EXPORT_DIRECTORY + MODEL_NAME + '.ckpt'
TEMP_DIRECTORY = 'temp_models/'
TEMP_CKPT_FILE = TEMP_DIRECTORY + MODEL_NAME + '.ckpt'

# MATLAB DICT KEYS
KEY_X_DATA_DICTIONARY = 'relevant_data'
KEY_Y_DATA_DICTIONARY = 'Y'

# IMAGE SHAPE/CHARACTERISTICS
DATA_WINDOW_SIZE = win_len
NUMBER_CLASSES = 5

DEFAULT_IMAGE_SHAPE = [TOTAL_DATA_CHANNELS, DATA_WINDOW_SIZE]
INPUT_IMAGE_SHAPE = [1, TOTAL_DATA_CHANNELS, DATA_WINDOW_SIZE]
SELECT_DATA_CHANNELS = np.asarray(range(1, TOTAL_DATA_CHANNELS + 1))
NUMBER_DATA_CHANNELS = SELECT_DATA_CHANNELS.shape[0]  # Selects first int in shape

# FOR MODEL DESIGN
if w_sel == 0:
    TRAINING_TOTAL = 128000
else:
    TRAINING_TOTAL = 32000
TRAIN_BATCH_SIZE = 64
NUMBER_STEPS = TRAINING_TOTAL // TRAIN_BATCH_SIZE
TEST_BATCH_SIZE = 100
LEARNING_RATE = 1e-3  # 1e-4  # 'Step size' on n-D optimization plane

STRIDE_CONV2D_1 = [1, 1, 1, 1]
STRIDE_CONV2D_2 = [1, 1, 1, 1]
STRIDE_CONV2D_3 = [1, 1, 2, 1]
STRIDE_CONV2D_4 = [1, 1, 2, 1]

BIAS_VAR_CL1 = 32  # Number of kernel convolutions in h_conv1
BIAS_VAR_CL2 = 32  # Number of kernel convolutions in h_conv2
BIAS_VAR_CL3 = 64  # Number of kernel convolutions in h_conv2
BIAS_VAR_CL4 = 64  # Number of kernel convolutions in h_conv2

WEIGHT_VAR_CL1 = [4, 1, 1, BIAS_VAR_CL1]
WEIGHT_VAR_CL2 = [2, 2, BIAS_VAR_CL1, BIAS_VAR_CL2]
WEIGHT_VAR_CL3 = [1, 4, BIAS_VAR_CL2, BIAS_VAR_CL3]
WEIGHT_VAR_CL4 = [2, 2, BIAS_VAR_CL3, BIAS_VAR_CL4]

UNITS_FC_LAYER = 1024

BIAS_VAR_FC1 = [UNITS_FC_LAYER]

WEIGHT_VAR_FC_OUTPUT = [*BIAS_VAR_FC1, NUMBER_CLASSES]

BIAS_VAR_FC_OUTPUT = [NUMBER_CLASSES]

# Start Script Here:
if not os.path.exists(EXPORT_DIRECTORY):
    os.mkdir(EXPORT_DIRECTORY)
    # os.mkdir(TEMP_DIRECTORY)
input_node_name = 'input'
keep_prob_node_name = 'keep_prob'
output_node_name = 'output'


def get_activations_mat(layer, input_val, shape):
    units = sess.run(layer, feed_dict={x: np.reshape(input_val, shape, order='F'), keep_prob: 1.0})
    # print("units.shape: ", units.shape)
    return units


def get_all_activations(training_data, folder_name0):
    w_hconv1 = np.empty([0, *h_conv1_shape[1:]], np.float32)
    w_hconv2 = np.empty([0, *h_conv2_shape[1:]], np.float32)
    w_hconv3 = np.empty([0, *h_conv3_shape[1:]], np.float32)
    w_hconv4 = np.empty([0, *h_conv4_shape[1:]], np.float32)
    w_hconv4_flat = np.empty([0, h_conv_flat_shape[1]], np.float32)
    # w_hfc1 = np.empty([0, h_fc1_shape[1]], np.float32)
    w_hfc1_do = np.empty([0, h_fc1_drop_shape[1]], np.float32)
    w_y_out = np.empty([0, y_conv_shape[1]], np.float32)
    print('Getting all Activations: please wait... ')
    for it in range(0, training_data.shape[0]):
        if it % 100 == 0:
            print('Saved Sample #', it)
        sample = training_data[it]
        w_hconv1 = np.concatenate((w_hconv1, get_activations_mat(h_conv1, sample, INPUT_IMAGE_SHAPE)), axis=0)
        w_hconv2 = np.concatenate((w_hconv2, get_activations_mat(h_conv2, sample, INPUT_IMAGE_SHAPE)), axis=0)
        w_hconv3 = np.concatenate((w_hconv3, get_activations_mat(h_conv3, sample, INPUT_IMAGE_SHAPE)), axis=0)
        w_hconv4 = np.concatenate((w_hconv4, get_activations_mat(h_conv4, sample, INPUT_IMAGE_SHAPE)), axis=0)
        w_hconv4_flat = np.concatenate((w_hconv4_flat, get_activations_mat(h_conv_flat, sample, INPUT_IMAGE_SHAPE)),
                                       axis=0)
        # w_hfc1 = np.concatenate((w_hfc1, get_activations_mat(h_fc1, sample, INPUT_IMAGE_SHAPE)), axis=0)
        w_hfc1_do = np.concatenate((w_hfc1_do, get_activations_mat(h_fc1_drop, sample, INPUT_IMAGE_SHAPE)), axis=0)
        w_y_out = np.concatenate((w_y_out, get_activations_mat(y_conv, sample, INPUT_IMAGE_SHAPE)), axis=0)
        # Save all activations:
    fn_out = folder_name0 + 'all_activations.mat'
    savemat(fn_out, mdict={'input_sample': training_data, 'h_conv1': w_hconv1, 'h_conv2': w_hconv2,
                           'h_conv_flat': w_hconv4_flat, 'h_fc1_drop': w_hfc1_do,
                           'y_out': w_y_out})


# MODEL INPUT #
x = tf.placeholder(tf.float32, shape=[None, *DEFAULT_IMAGE_SHAPE], name=input_node_name)
keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
y = tf.placeholder(tf.float32, shape=[None, NUMBER_CLASSES])

x_input = tf.reshape(x, [-1, *DEFAULT_IMAGE_SHAPE, 1])

# first convolution and pooling
W_conv1 = tfs.weight(WEIGHT_VAR_CL1)
b_conv1 = tfs.bias([BIAS_VAR_CL1])
h_conv1 = tfs.elu_conv(x_input, W_conv1, b_conv1, STRIDE_CONV2D_1)

# second convolution and pooling
W_conv2 = tfs.weight(WEIGHT_VAR_CL2)
b_conv2 = tfs.bias([BIAS_VAR_CL2])
h_conv2 = tfs.elu_conv(h_conv1, W_conv2, b_conv2, STRIDE_CONV2D_2)

h_mp_1 = tfs.max_pool(h_conv2, [1, 2, 2, 1], [1, 1, 1, 1])

W_conv3 = tfs.weight(WEIGHT_VAR_CL3)
b_conv3 = tfs.bias([BIAS_VAR_CL3])
h_conv3 = tfs.elu_conv(h_mp_1, W_conv3, b_conv3, STRIDE_CONV2D_3)

W_conv4 = tfs.weight(WEIGHT_VAR_CL4)
b_conv4 = tfs.bias([BIAS_VAR_CL4])
h_conv4 = tfs.elu_conv(h_conv3, W_conv4, b_conv4, STRIDE_CONV2D_4)

h_mp_2 = tfs.max_pool(h_conv4, [1, 2, 2, 1], [1, 1, 2, 1])

# the input should be shaped/flattened
h_conv_flat, layer_shape = tfs.flatten(h_mp_2)

# fully connected layer1,the shape of the patch should be defined
W_fc1 = tfs.weight([layer_shape, UNITS_FC_LAYER])
b_fc1 = tfs.bias(BIAS_VAR_FC1)

h_fc1_drop = tfs.fully_connect_relu_dropout(h_conv_flat, W_fc1, b_fc1, keep_prob)
# h_fc1_drop = tfs.fully_connect_elu_dropout(h_conv_flat, W_fc1, b_fc1, keep_prob)
# h_fc1_drop = tfs.fully_connect_leakyrelu_dropout(h_conv_flat, W_fc1, b_fc1, keep_prob)

# weight and bias of the output layer
W_fco = tfs.weight(WEIGHT_VAR_FC_OUTPUT)
b_fco = tfs.bias(BIAS_VAR_FC_OUTPUT)

y_conv = tfs.connect(h_fc1_drop, W_fco, b_fco)

outputs = tf.nn.softmax(y_conv, name=output_node_name)

prediction = tf.argmax(outputs, 1)

# training and reducing the cost/loss function
cross_entropy = tfs.loss_layer(y, y_conv)
train_step = tfs.train(LEARNING_RATE, cross_entropy)
correct_prediction = tfs.check_prediction(y, outputs)
accuracy = tfs.get_accuracy(correct_prediction)  # Float 32

# merge
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()  # Initialize tf Saver

# Load Data:
print('Train Folder Path: ', TRAINING_FOLDER_PATH)
x_data, y_data = tfs.load_data(TRAINING_FOLDER_PATH, DEFAULT_IMAGE_SHAPE, KEY_X_DATA_DICTIONARY, KEY_Y_DATA_DICTIONARY)
x_val_data, y_val_data = tfs.load_data(TEST_FOLDER_PATH,
                                       DEFAULT_IMAGE_SHAPE, KEY_X_DATA_DICTIONARY, KEY_Y_DATA_DICTIONARY)
# Split training set:
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.75, random_state=1)

# TRAIN ROUTINE #
init_op = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

val_step = 0
with tf.Session(config=config) as sess:
    sess.run(init_op)
    # TODO save/restore checkpoints

    x_0 = np.zeros(INPUT_IMAGE_SHAPE, dtype=np.float32)
    print("Model Dimensions: ")
    h_conv1_shape = sess.run(h_conv1, feed_dict={x: x_0, keep_prob: 1.0}).shape
    h_conv2_shape = sess.run(h_conv2, feed_dict={x: x_0, keep_prob: 1.0}).shape
    h_mp_1_shape = sess.run(h_mp_1, feed_dict={x: x_0, keep_prob: 1.0}).shape
    h_conv3_shape = sess.run(h_conv3, feed_dict={x: x_0, keep_prob: 1.0}).shape
    h_conv4_shape = sess.run(h_conv4, feed_dict={x: x_0, keep_prob: 1.0}).shape
    h_mp_2_shape = sess.run(h_mp_2, feed_dict={x: x_0, keep_prob: 1.0}).shape
    h_conv_flat_shape = sess.run(h_conv_flat, feed_dict={x: x_0, keep_prob: 1.0}).shape
    h_fc1_drop_shape = sess.run(h_fc1_drop, feed_dict={x: x_0, keep_prob: 1.0}).shape
    y_conv_shape = sess.run(y_conv, feed_dict={x: x_0, keep_prob: 1.0}).shape
    print("h_conv1: ", h_conv1_shape)
    print("h_conv2: ", h_conv2_shape)
    print("h_mp1: ", h_mp_1_shape)
    print("h_conv3: ", h_conv3_shape)
    print("h_conv4: ", h_conv4_shape)
    print("h_mp2: ", h_mp_2_shape)
    print("h_conv_flat: ", h_conv_flat_shape)
    print("h_fc1_drop: ", h_fc1_drop_shape)
    print("y_conv: ", y_conv_shape)

    # save model as pbtxt:
    tf.train.write_graph(sess.graph_def, EXPORT_DIRECTORY, MODEL_NAME + '.pbtxt', True)
    total_val_steps = NUMBER_STEPS // 20
    val_accuracy_array = np.zeros([total_val_steps, 2], dtype=np.float32)
    for i in range(NUMBER_STEPS):
        offset = (i * TRAIN_BATCH_SIZE) % (x_train.shape[0] - TRAIN_BATCH_SIZE)
        batch_x_train = x_train[offset:(offset + TRAIN_BATCH_SIZE)]
        batch_y_train = y_train[offset:(offset + TRAIN_BATCH_SIZE)]
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_x_train, y: batch_y_train, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        if i % 20 == 0:
            # Calculate batch loss and accuracy
            offset = (val_step * TEST_BATCH_SIZE) % (x_test.shape[0] - TEST_BATCH_SIZE)
            batch_x_val = x_test[offset:(offset + TEST_BATCH_SIZE), :, :]
            batch_y_val = y_test[offset:(offset + TEST_BATCH_SIZE), :]
            val_accuracy = accuracy.eval(feed_dict={x: batch_x_val, y: batch_y_val, keep_prob: 1.0})
            print("Validation step %d, validation accuracy %g" % (val_step, val_accuracy))
            val_accuracy_array[val_step, 0] = (1 + val_step) * 20 * TRAIN_BATCH_SIZE
            val_accuracy_array[val_step, 1] = val_accuracy
            val_step += 1

        train_step.run(feed_dict={x: batch_x_train, y: batch_y_train, keep_prob: 0.5})

    # Run test data (entire set) to see accuracy.
    test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.0})  # original
    print("\n Testing Accuracy:", test_accuracy, "\n\n")

    # Holdout Validation Accuracy:
    print("\n Holdout Validation:", sess.run(accuracy, feed_dict={x: x_val_data, y: y_val_data, keep_prob: 1.0}))

    y_val_tf = np.zeros([x_val_data.shape[0]], dtype=np.int32)
    predictions = np.zeros([x_val_data.shape[0]], dtype=np.int32)
    for i in range(0, x_val_data.shape[0]):
        predictions[i] = sess.run(prediction,
                                  feed_dict={x: x_val_data[i].reshape(INPUT_IMAGE_SHAPE),
                                             y: y_val_data[i].reshape([1, NUMBER_CLASSES]), keep_prob: 1.0})
        for c in range(0, NUMBER_CLASSES):
            if y_val_data[i][c]:
                y_val_tf[i] = c

    tf_confusion_matrix = tf.confusion_matrix(labels=y_val_tf, predictions=predictions, num_classes=NUMBER_CLASSES)
    print(tf.Tensor.eval(tf_confusion_matrix, feed_dict=None, session=None))  # 'Confusion Matrix: \n\n',

    # Get one sample and see what it outputs (Activations?) ?

    ws.Beep(900, 1000)
    # Extract weights of following layers
    user_input = input('Save all activations?')
    if user_input == "1" or user_input.lower() == "y":
        feature_map_folder_name = \
            EXPORT_DIRECTORY + 'S' + str(subject_number) + \
            'feature_maps_' + TIMESTAMP_START + '_wlen' + str(DATA_WINDOW_SIZE) + '/'
        os.makedirs(feature_map_folder_name)
        get_all_activations(x_val_data, feature_map_folder_name)
        tfs.save_statistics(feature_map_folder_name, val_accuracy_array)
    #
    user_input = input('Export Current Model?')
    if user_input == "1" or user_input.lower() == "y":
        saver.save(sess, CHECKPOINT_FILE)
        # export_model([input_node_name, keep_prob_node_name], output_node_name)
        tfs.export_model([input_node_name, keep_prob_node_name], output_node_name, EXPORT_DIRECTORY, MODEL_NAME)
