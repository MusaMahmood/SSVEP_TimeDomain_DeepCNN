# MUSA MAHMOOD - Copyright 2018
# Python 3.6.3
# TF 1.5.0

# IMPORTS:
import tensorflow as tf
import os as os
import tf_shared as tfs
import numpy as np
from sklearn.model_selection import train_test_split

EXPORT_DIRECTORY = 'model_exports/'
wlens = [128, 192, 256, 384, 512]
win_len = wlens[0]
subject_number = 99
TRAINING_FOLDER = r'' + 'time_domain_hpf' + '/S' + str(subject_number) + '/' + 'w' + str(win_len)
input_shape = [2, win_len]  # for single sample reshape(x, [1, *input_shape])
NUMBER_CLASSES = 5

# LOAD DATA:
x_tt, y_tt = tfs.load_data(TRAINING_FOLDER, input_shape, key_x='relevant_data', key_y='Y')
x_train, x_test, y_train, y_test = train_test_split(x_tt, y_tt, train_size=0.75, random_state=1)
x_val, y_val = tfs.load_data(TRAINING_FOLDER + '/v', input_shape, key_x='relevant_data', key_y='Y')
x_val2, y_val2 = tfs.load_data(TRAINING_FOLDER + '/v2', input_shape, key_x='relevant_data', key_y='Y')
# Initialize CNN Components
NUM_LAYERS = 1  # Default
N_FILTERS = [25, 2, 5, 5, 5, 5, 5, 5]  # Number of filters for 8 layers
Str_X = [1, 1, 1, 1, 1, 1, 1, 1]
Str_Y = [1, 1, 1, 1, 1, 1, 1, 1]
W_x = [20, 2, 2, 2, 2, 2, 2, 2]  # Weights in x-axis
W_y = [2, 2, 2, 2, 2, 2, 2, 2]  # Weights in y-axis, kernel size = [x, y]
Alphas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Parametric ReLU values for Conv Layers
Alpha_fc = 0.01  # If using parametric ReLU in fully connected layer.
do = 'dropout'
keep_prob_feed = 0.5
train_steps = 3000
LR_EXP = 3
LR_COEFF = 1
learn_rate = float(LR_COEFF) * float(10.0 ** (-float(LR_EXP)))
activation_fc = 'relu'
activation = 'parametricrelu'
UNITS_FC = 1024
Model_description = tfs.get_model_description(NUM_LAYERS, activation, do, keep_prob_feed, UNITS_FC,
                                              activation_fc, LR_COEFF, LR_EXP, N_FILTERS, Alphas[0:NUM_LAYERS])

if not os.path.exists(EXPORT_DIRECTORY):
    os.mkdir(EXPORT_DIRECTORY)
input_node_name = 'input'
keep_prob_node_name = 'keep_prob'
output_node_name = 'output'

# 1. Load Model Placeholders:
x, y, keep_prob = tfs.placeholders(input_shape, NUMBER_CLASSES, input_node_name, keep_prob_node_name)
# 2. Reshape x input tensor:
x_input = tfs.reshape_input(x, input_shape)
# 3. Add First Convolution Layers: (Collect, layer, weights and biases)
hwb_list = [tfs.conv_layer_named(x_input, [W_x[0], W_y[0]], 1, N_FILTERS[0], [Str_X[0], Str_Y[0]], 'hc1', 'bc1',
                                 activation, Alphas[0])]
# 4. Add Additional Conv Layers as Needed:
for i in range(1, NUM_LAYERS):
    hwb_list.append(tfs.conv_layer(hwb_list[i - 1][0], [W_x[i], W_y[i]], N_FILTERS[i - 1], N_FILTERS[i],
                                   [Str_X[i], Str_Y[i]], activation, Alphas[i]))
# 4.1 Extract hidden layers to list
h = [i[0] for i in hwb_list]
# 5. Flatten and Fully Connect:
h_flat, flat_shape = tfs.flatten(h[NUM_LAYERS - 1])
h_fc = tfs.fully_connect_layer(h_flat, [flat_shape, UNITS_FC], [UNITS_FC], do, keep_prob, activation_fc, Alpha_fc)
# 6. Connect to form output:
y_conv = tfs.output_layer(h_fc, [UNITS_FC, NUMBER_CLASSES], [NUMBER_CLASSES])
# 7. Compute loss/cross-entropy for training step (using Adam gradient optimization):
train_step = tfs.train_loss(y, y_conv, learn_rate)
# 8. Output Node, compute accuracy [Softmax outputs, output int, accuracy]
outputs, prediction, accuracy = tfs.get_outputs(y, y_conv, output_node_name)
# 9. Initialize tensorflow for training & evaluation:
saver, init_op, config = tfs.tf_initialize()

# Enter Training Routine:
with tf.Session(config=config) as sess:
    sess.run(init_op)
    # Print Model Information Before Starting.
    model_dims = tfs.get_model_dimensions(h, h_flat, h_fc, y_conv, NUM_LAYERS)
    filter_dims = tfs.get_filter_dimensions(W_x, W_y, Str_X, Str_Y, Alphas, NUM_LAYERS)
    print(model_dims, '\n', filter_dims)
    # Save model as pbtxt:
    tf.train.write_graph(sess.graph_def, EXPORT_DIRECTORY, Model_description + '.pbtxt', True)
    start_time_ms = tfs.current_time_ms()
    # Train Model:
    val_accuracy_rate = tfs.train(x, y, keep_prob, accuracy, train_step, x_train, y_train, x_test, y_test,
                                  keep_prob_feed, train_steps)
    elapsed_time_ms = (tfs.current_time_ms() - start_time_ms)
    # Test Accuracy: (Test/Train Split)
    tt_acc = tfs.test(sess, x, y, accuracy, x_test, y_test, keep_prob, test_type='Train-Split')
    # Validation Accuracy:
    val_acc = tfs.test(sess, x, y, accuracy, x_val, y_val, keep_prob)
    # Confusion Matrix:
    conf_mat_v1 = tfs.confusion_matrix_test(sess, x, y, keep_prob, prediction, [1, *input_shape], x_val, y_val, NUMBER_CLASSES)
    val_acc2 = 0.0
    conf_mat_v2 = np.zeros([5, 5], dtype=np.int32)
    if x_val2.shape[0] > 1:
        val_acc2 = tfs.test(sess, x, y, accuracy, x_val2, y_val2, keep_prob, test_type='Validation-Subject')
        conf_mat_v2 = tfs.confusion_matrix_test(sess, x, y, keep_prob, prediction, [1, *input_shape], x_val2, y_val2, NUMBER_CLASSES)
    else:
        print('No validation file found!')
    print('Elapsed Time (ms): ', elapsed_time_ms)

    tfs.beep()
    # Save Statistics:
    output_folder_name = EXPORT_DIRECTORY + 'S' + str(subject_number) + '/wlen' + str(win_len) + '/'
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    stat_fn = 'stats_' + Model_description + '.mat'
    user_input = input('Export Current Model?')
    if user_input == "1" or user_input.lower() == "y":
        tfs.get_trained_vars(sess, output_folder_name + Model_description)
        # tfs.save_statistics_v2(output_folder_name, val_accuracy_rate, Model_description, model_dims + filter_dims,
        #                        elapsed_time_ms, val_acc, val_acc2, stat_fn)
        tfs.save_statistics_v3(output_folder_name, val_accuracy_rate, Model_description, model_dims + filter_dims,
                               elapsed_time_ms, val_acc, conf_mat_v1, val_acc2, conf_mat_v2, stat_fn)
        tfs.get_all_activations(sess, x, keep_prob, [1, *input_shape], x_val, y_val,
                                output_folder_name + Model_description, h, h_flat, h_fc, y_conv)
    #     CHECKPOINT_FILE = EXPORT_DIRECTORY + Model_description + '.ckpt'
    #     saver.save(sess, CHECKPOINT_FILE)
    #     tfs.export_model([input_node_name, keep_prob_node_name], output_node_name,
    #       EXPORT_DIRECTORY, Model_description)
