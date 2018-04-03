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
from sklearn.model_selection import train_test_split

# CONSTANTS:
TIMESTAMP_START = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H.%M.%S')
wlens = [128, 192, 256, 384, 512]
w_sel = 0  # 0 -> 4
win_len = wlens[w_sel]
electrodes = ''
input_data_descriptor = 'time_domain_hpf_new'  # FIXED + TRUNCATED
subject_number = 99
subject_folder = '/S' + str(subject_number)
folder_name = ''

TOTAL_DATA_CHANNELS = 2
TRAINING_FOLDER_PATH = r'' + input_data_descriptor + subject_folder + '/' + 'w' + str(win_len)
TEST_FOLDER_PATH = TRAINING_FOLDER_PATH + '/v'
EXPORT_DIRECTORY = 'model_exports/'
MODEL_NAME = 'ssvep_net_2ch' + '_S' + str(subject_number) + '_' + input_data_descriptor + '_wlen' + str(win_len)
CHECKPOINT_FILE = EXPORT_DIRECTORY + MODEL_NAME + '.ckpt'

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

# INFORMATION!
"""
    Options for activation are :
    'relu'
    'elu'
    'leakyrelu'
    'parametricrelu'
"""
activation = 'parametricrelu'
fc_activation = 'relu'
alpha_fc = 0.01
alpha_conv = 0.5

do = "dropout"  # dropout or no-dropout
# do = "no-dropout"  # dropout or no-dropout

# FOR MODEL DESIGN
TRAINING_TOTAL = 256000
TRAIN_BATCH_SIZE = 64
NUMBER_STEPS = TRAINING_TOTAL // TRAIN_BATCH_SIZE
TEST_BATCH_SIZE = 100
LR_EXP = 3
LR_COEFF = 1.0
LEARNING_RATE = LR_COEFF * float(10.0 ** (-float(LR_EXP)))
NUM_LAYERS = 4

STRIDE_CONV2D_1 = [1, 1, 1, 1]
STRIDE_CONV2D_2 = [1, 1, 1, 1]
STRIDE_CONV2D_3 = [1, 1, 1, 1]
STRIDE_CONV2D_4 = [1, 1, 1, 1]

BIAS_VAR_CL1 = 32  # Number of kernel convolutions in h_conv1
BIAS_VAR_CL2 = 32  # Number of kernel convolutions in h_conv2
BIAS_VAR_CL3 = 64  # Number of kernel convolutions in h_conv2
BIAS_VAR_CL4 = 64  # Number of kernel convolutions in h_conv2

WEIGHT_VAR_CL1 = [4, 4, 1, BIAS_VAR_CL1]
WEIGHT_VAR_CL2 = [4, 4, BIAS_VAR_CL1, BIAS_VAR_CL2]
WEIGHT_VAR_CL3 = [4, 4, BIAS_VAR_CL2, BIAS_VAR_CL3]
WEIGHT_VAR_CL4 = [4, 4, BIAS_VAR_CL3, BIAS_VAR_CL4]

UNITS_FC_LAYER = 1024

BIAS_VAR_FC1 = [UNITS_FC_LAYER]

WEIGHT_VAR_FC_OUTPUT = [*BIAS_VAR_FC1, NUMBER_CLASSES]

BIAS_VAR_FC_OUTPUT = [NUMBER_CLASSES]

if activation == 'parametricrelu' and fc_activation == 'parametricrelu':
    MODEL_DESCRIPTION = 'CNN-' + str(NUM_LAYERS) + "-a." + activation + '.alpha-' + str(alpha_conv) + '-' + do \
                        + '-fc.' + fc_activation + '.alpha-' + str(alpha_fc) + '-lr.1e-' + str(LR_EXP)
elif activation == 'parametricrelu':
    MODEL_DESCRIPTION = 'CNN-' + str(NUM_LAYERS) + "-a." + activation + '.alpha-' + str(alpha_conv) + '-' + do \
                        + '-fc.' + fc_activation + '-lr.1e-' + str(LR_EXP)
elif fc_activation == 'parametricrelu':
    MODEL_DESCRIPTION = 'CNN-' + str(NUM_LAYERS) + "-a." + activation + '-' + do + '-fc.' \
                        + fc_activation + '.alpha-' + str(alpha_fc) + '-lr.1e-' + str(LR_EXP)
else:
    MODEL_DESCRIPTION = 'CNN-' + str(NUM_LAYERS) + "-a." + activation + '-' + do + '-fc.' + fc_activation \
                        + '-lr.1e-' + str(LR_EXP)

# Start Script Here:
if not os.path.exists(EXPORT_DIRECTORY):
    os.mkdir(EXPORT_DIRECTORY)
input_node_name = 'input'
keep_prob_node_name = 'keep_prob'
output_node_name = 'output'

# MODEL INPUT #
x = tf.placeholder(tf.float32, shape=[None, *DEFAULT_IMAGE_SHAPE], name=input_node_name)
keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
y = tf.placeholder(tf.float32, shape=[None, NUMBER_CLASSES])

# 1- Reshape Input to Default [n, x, y, 1] dimensions
x_input = tf.reshape(x, [-1, *DEFAULT_IMAGE_SHAPE, 1])

# first convolution and pooling
W_conv1 = tfs.weight(WEIGHT_VAR_CL1)
b_conv1 = tfs.bias([BIAS_VAR_CL1])
h_conv1 = tfs.conv(x_input, W_conv1, b_conv1, STRIDE_CONV2D_1, activation=activation, alpha=alpha_conv)

# second convolution and pooling
W_conv2 = tfs.weight(WEIGHT_VAR_CL2)
b_conv2 = tfs.bias([BIAS_VAR_CL2])
h_conv2 = tfs.conv(h_conv1, W_conv2, b_conv2, STRIDE_CONV2D_2, activation=activation, alpha=alpha_conv)

W_conv3 = tfs.weight(WEIGHT_VAR_CL3)
b_conv3 = tfs.bias([BIAS_VAR_CL3])
h_conv3 = tfs.conv(h_conv2, W_conv3, b_conv3, STRIDE_CONV2D_3, activation=activation, alpha=alpha_conv)

W_conv4 = tfs.weight(WEIGHT_VAR_CL4)
b_conv4 = tfs.bias([BIAS_VAR_CL4])
h_conv4 = tfs.conv(h_conv3, W_conv4, b_conv4, STRIDE_CONV2D_4, activation=activation, alpha=alpha_conv)

# The input should be shaped/flattened
h_flat, h_flat_shape = tfs.flatten(h_conv4)

# fully connected layer,the shape of the patch should be defined
W_fc1 = tfs.weight([h_flat_shape, UNITS_FC_LAYER])
b_fc1 = tfs.bias(BIAS_VAR_FC1)
if do == "no-dropout":
    h_fc1 = tfs.fully_connect(h_flat, W_fc1, b_fc1, activation=fc_activation, alpha=alpha_fc)
else:
    h_fc1 = tfs.fully_connect_with_dropout(h_flat, W_fc1, b_fc1, keep_prob, activation=fc_activation, alpha=alpha_fc)

# weight and bias of the output layer
W_fco = tfs.weight(WEIGHT_VAR_FC_OUTPUT)
b_fco = tfs.bias(BIAS_VAR_FC_OUTPUT)
y_conv = tfs.connect(h_fc1, W_fco, b_fco)

# training and reducing the cost/loss function
cross_entropy = tfs.loss_layer_v2(y, y_conv)
train_step = tfs.train(LEARNING_RATE, cross_entropy)
# Output Node and Prediction; Correctness, and Accuracy
outputs = tf.nn.softmax(y_conv, name=output_node_name)
prediction_check, prediction = tfs.check_prediction(y, outputs)
accuracy = tfs.get_accuracy(prediction_check)

# Load Data:
print('Train Folder Path: ', TRAINING_FOLDER_PATH)
x_data, y_data = tfs.load_data(TRAINING_FOLDER_PATH, DEFAULT_IMAGE_SHAPE, KEY_X_DATA_DICTIONARY, KEY_Y_DATA_DICTIONARY)
x_val_data, y_val_data = tfs.load_data(TEST_FOLDER_PATH,
                                       DEFAULT_IMAGE_SHAPE, KEY_X_DATA_DICTIONARY, KEY_Y_DATA_DICTIONARY)
# Split training set:
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.75, random_state=1)

# Merges all summaries collected in the default graph.
merged_summary_op = tf.summary.merge_all()
saver = tf.train.Saver()  # Initialize tf Saver for model export
init_op = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

val_step = 0
# TRAIN ROUTINE #
with tf.Session(config=config) as sess:
    sess.run(init_op)
    # TODO save/restore checkpoints

    x_0 = np.zeros(INPUT_IMAGE_SHAPE, dtype=np.float32)
    print("Model Dimensions: ")
    print("h_conv1: ", tfs.get_tensor_shape_tuple(h_conv1))
    # print("h_conv2: ", tfs.get_tensor_shape_tuple(h_conv2))
    # print("h_conv3: ", tfs.get_tensor_shape_tuple(h_conv3))
    # print("h_conv4: ", tfs.get_tensor_shape_tuple(h_conv4))
    print("h_flat: ", tfs.get_tensor_shape_tuple(h_flat))
    print("h_fc1: ", tfs.get_tensor_shape_tuple(h_fc1))
    print("y_conv: ", tfs.get_tensor_shape_tuple(y_conv))
    print(":")
    print("Filter Dimensions:")
    print("h_c1_filt: ", WEIGHT_VAR_CL1[0:2], " stride: ", STRIDE_CONV2D_1[1:3], " alpha=", alpha_conv)
    print("h_c2_filt: ", WEIGHT_VAR_CL2[0:2], " stride: ", STRIDE_CONV2D_2[1:3], " alpha=", alpha_conv)
    print("h_c3_filt: ", WEIGHT_VAR_CL3[0:2], " stride: ", STRIDE_CONV2D_3[1:3], " alpha=", alpha_conv)
    print("h_c4_filt: ", WEIGHT_VAR_CL4[0:2], " stride: ", STRIDE_CONV2D_4[1:3], " alpha=", alpha_conv)
    print(":")

    # save model as pbtxt:
    tf.train.write_graph(sess.graph_def, EXPORT_DIRECTORY, MODEL_NAME + '.pbtxt', True)
    total_val_steps = NUMBER_STEPS // 20
    val_accuracy_array = np.zeros([total_val_steps, 2], dtype=np.float32)
    # START TRAINING TIMER
    START_TIME_MS = tfs.current_time_ms()
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
    FINISH_TIME_MS = tfs.current_time_ms()  # FINISH TRAINING TIMER
    # Run test data (entire set) to see accuracy.
    test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.0})  # original
    print("\n Testing Accuracy:", test_accuracy, "\n\n")

    # Holdout Validation Accuracy:
    result_string = "\n Holdout Validation:" \
                    + str(sess.run(accuracy, feed_dict={x: x_val_data, y: y_val_data, keep_prob: 1.0}))
    print(result_string)
    print('\n elapsed time: ', str((FINISH_TIME_MS - START_TIME_MS) / 1000.0) + ' s')
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

    ws.Beep(900, 1000)
    # Save Data:
    INFO = "h_c1_filt: " + str(WEIGHT_VAR_CL1[0:2]) + " stride: " + str(STRIDE_CONV2D_1[1:3]) + " alpha=" + str(
        alpha_conv) + "\n" + \
           "h_c2_filt: " + str(WEIGHT_VAR_CL2[0:2]) + " stride: " + str(STRIDE_CONV2D_2[1:3]) + " alpha=" + str(
        alpha_conv) + "\n" + \
           "h_c3_filt: " + str(WEIGHT_VAR_CL3[0:2]) + " stride: " + str(STRIDE_CONV2D_3[1:3]) + " alpha=" + str(
        alpha_conv) + "\n" + \
           "h_c4_filt: " + str(WEIGHT_VAR_CL4[0:2]) + " stride: " + str(STRIDE_CONV2D_4[1:3]) + " alpha=" + str(
        alpha_conv) + "\n" + "alphafc=" + str(alpha_fc) + '\n' + result_string + '\n' + \
           'elapsed time (ms):' + str(FINISH_TIME_MS - START_TIME_MS)
    output_folder_name = EXPORT_DIRECTORY + 'S' + str(subject_number) + '_wlen' + str(DATA_WINDOW_SIZE) + '/'
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    stat_fn = 'stats_' + MODEL_DESCRIPTION + '.mat'
    tfs.save_statistics(output_folder_name, val_accuracy_array, MODEL_DESCRIPTION, INFO, stat_fn)

    # tfs.get_all_activations_4layer(sess, x, keep_prob, INPUT_IMAGE_SHAPE, x_val_data, output_folder_name, h_conv1,
    #                                h_conv2, h_conv3, h_conv4, h_flat, h_fc1, y_conv)
    # user_input = input('Export Current Model?')
    # if user_input == "1" or user_input.lower() == "y":
    #     saver.save(sess, CHECKPOINT_FILE)
    #     # export_model([input_node_name, keep_prob_node_name], output_node_name)
    #     tfs.export_model([input_node_name, keep_prob_node_name], output_node_name, EXPORT_DIRECTORY, MODEL_NAME)
