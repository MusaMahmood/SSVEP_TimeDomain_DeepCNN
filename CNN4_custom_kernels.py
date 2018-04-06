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
descriptor = 'time_domain_hpf_new'  # FIXED + TRUNCATED
subject_number = 8
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
MATLAB_DICT_KEY_X = 'relevant_data'
MATLAB_DICT_KEY_Y = 'Y'

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
alpha_c1 = 0.1
STRIDE_CONV2D_2 = [1, 1, 1, 1]
alpha_c2 = 0.2
STRIDE_CONV2D_3 = [1, 1, 2, 1]
alpha_c3 = 0.3
STRIDE_CONV2D_4 = [1, 1, 2, 1]
alpha_c4 = 0.4

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
input_node_name = 'input'
keep_prob_node_name = 'keep_prob'
output_node_name = 'output'

# MODEL INPUT #
x = tf.placeholder(tf.float32, shape=[None, *DEFAULT_IMAGE_SHAPE], name=input_node_name)
keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
y = tf.placeholder(tf.float32, shape=[None, NUMBER_CLASSES])

x_input = tf.reshape(x, [-1, *DEFAULT_IMAGE_SHAPE, 1])

# first convolution and pooling
W_conv1 = tfs.weight(WEIGHT_VAR_CL1)
b_conv1 = tfs.bias([BIAS_VAR_CL1])
h_conv1 = tfs.leaky_conv(x_input, W_conv1, b_conv1, STRIDE_CONV2D_1, alpha=alpha_c1)

# second convolution and pooling
W_conv2 = tfs.weight(WEIGHT_VAR_CL2)
b_conv2 = tfs.bias([BIAS_VAR_CL2])
h_conv2 = tfs.leaky_conv(h_conv1, W_conv2, b_conv2, STRIDE_CONV2D_2, alpha=alpha_c2)

W_conv3 = tfs.weight(WEIGHT_VAR_CL3)
b_conv3 = tfs.bias([BIAS_VAR_CL3])
h_conv3 = tfs.leaky_conv(h_conv2, W_conv3, b_conv3, STRIDE_CONV2D_3, alpha=alpha_c3)

W_conv4 = tfs.weight(WEIGHT_VAR_CL4)
b_conv4 = tfs.bias([BIAS_VAR_CL4])
h_conv4 = tfs.leaky_conv(h_conv3, W_conv4, b_conv4, STRIDE_CONV2D_4, alpha=alpha_c4)

# The input should be shaped/flattened
h_conv_flat, layer_shape = tfs.flatten(h_conv4)

# fully connected layer1,the shape of the patch should be defined
W_fc1 = tfs.weight([layer_shape, UNITS_FC_LAYER])
b_fc1 = tfs.bias(BIAS_VAR_FC1)
h_fc1_drop = tfs.fully_connect_with_dropout(h_conv_flat, W_fc1, b_fc1, keep_prob, activation='relu')

# weight and bias of the output layer
W_fco = tfs.weight(WEIGHT_VAR_FC_OUTPUT)
b_fco = tfs.bias(BIAS_VAR_FC_OUTPUT)
y_conv = tfs.connect(h_fc1_drop, W_fco, b_fco)

# training and reducing the cost/loss function
cross_entropy = tfs.loss_layer_v2(y, y_conv)
train_step = tfs.train_optimize(LEARNING_RATE, cross_entropy)
# Output Node and Prediction; is it correct, and accuracy
outputs = tf.nn.softmax(y_conv, name=output_node_name)
prediction_check, prediction = tfs.check_prediction(y, outputs)
accuracy = tfs.get_accuracy(prediction_check)  # Float 32

# Load Data:
print('Train Folder Path: ', TRAINING_FOLDER_PATH)
x_data, y_data = tfs.load_data(TRAINING_FOLDER_PATH, DEFAULT_IMAGE_SHAPE, MATLAB_DICT_KEY_X, MATLAB_DICT_KEY_Y)
x_val_data, y_val_data = tfs.load_data(TEST_FOLDER_PATH, DEFAULT_IMAGE_SHAPE, MATLAB_DICT_KEY_X, MATLAB_DICT_KEY_Y)
# Split training set:
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.75, random_state=1)

# TRAIN ROUTINE #
# Merges all summaries collected in the default graph.
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()  # Initialize tf Saver for model export

init_op = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

val_step = 0
with tf.Session(config=config) as sess:
    sess.run(init_op)
    # TODO save/restore checkpoints

    x_0 = np.zeros(INPUT_IMAGE_SHAPE, dtype=np.float32)
    print("Model Dimensions: ")
    print("h_conv1: ", tfs.get_tensor_shape_tuple(h_conv1))
    print("h_conv2: ", tfs.get_tensor_shape_tuple(h_conv2))
    print("h_conv3: ", tfs.get_tensor_shape_tuple(h_conv3))
    print("h_conv4: ", tfs.get_tensor_shape_tuple(h_conv4))
    print("h_flat: ", tfs.get_tensor_shape_tuple(h_conv_flat))
    print("h_fc1: ", tfs.get_tensor_shape_tuple(h_fc1_drop))
    print("y_conv: ", tfs.get_tensor_shape_tuple(y_conv))
    print(":")
    print("Filter Dimensions:")
    print("h_c1_filt: ", WEIGHT_VAR_CL1[0:2], " stride: ", STRIDE_CONV2D_1[1:3], " alpha=", alpha_c1)
    print("h_c2_filt: ", WEIGHT_VAR_CL2[0:2], " stride: ", STRIDE_CONV2D_2[1:3], " alpha=", alpha_c2)
    print("h_c3_filt: ", WEIGHT_VAR_CL3[0:2], " stride: ", STRIDE_CONV2D_3[1:3], " alpha=", alpha_c3)
    print("h_c4_filt: ", WEIGHT_VAR_CL4[0:2], " stride: ", STRIDE_CONV2D_4[1:3], " alpha=", alpha_c4)

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

    ws.Beep(900, 1000)
    # Extract weights of following layers
    user_input = input('Save all activations?')
    if user_input == "1" or user_input.lower() == "y":
        output_folder_name = \
            EXPORT_DIRECTORY + 'S' + str(subject_number) + \
            'feature_maps_' + TIMESTAMP_START + '_wlen' + str(DATA_WINDOW_SIZE) + '/'
        os.makedirs(output_folder_name)
        tfs.get_all_activations_4layer(sess, x, keep_prob, INPUT_IMAGE_SHAPE, x_val_data, output_folder_name, h_conv1,
                                       h_conv2, h_conv3, h_conv4, h_conv_flat, h_fc1_drop, y_conv)
        tfs.save_statistics(output_folder_name, val_accuracy_array)

    user_input = input('Export Current Model?')
    if user_input == "1" or user_input.lower() == "y":
        saver.save(sess, CHECKPOINT_FILE)
        # export_model([input_node_name, keep_prob_node_name], output_node_name)
        tfs.export_model([input_node_name, keep_prob_node_name], output_node_name, EXPORT_DIRECTORY, MODEL_NAME)
