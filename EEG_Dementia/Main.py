import tensorflow as tf
import numpy as np
import EEG_Dementia.DataTransform as dt
import os.path



# Data description
# 2 channel EEG data. 1min, 256 s/s -> 15360 * 2ch
# reshape to [60*2, 256]
# ch1 0-1s data [n, 256]
# ch2 0-1s data
# ch1 1-2s data
# ch2 1-2s data
# Y - classifying to Normal and Dementia

# load data
# X_Normal - 28 subjects, 60 window * 2 channel, 256 -> 1sec
# X_Dementia - 45 subjects, 60 window * 2 channel, 256 -> 1sec
# check if data file is exist and load
if (os.path.isfile('Data/Data_Normal.npy')) and (os.path.isfile('Data/Data_Dementia.npy')):
    X_Normal = np.load('Data/Data_Normal.npy')
    X_Dementia = np.load('Data/Data_Dementia.npy')
else:
    print("create data")
    dt.create_data(28, 45)  # X_Normal size, X_Dementia size
    X_Normal = np.load('Data/Data_Normal.npy')
    X_Dementia = np.load('Data/Data_Dementia.npy')

# normal data, dementia data, testing size for normal, testing size for dementia
X_training, X_testing = dt.separate_data(X_Normal, X_Dementia, 6, 9)
X_training = np.expand_dims(X_training, axis=4)
X_testing = np.expand_dims(X_testing, axis=4)
print(np.shape(X_training), np.shape(X_testing))


# Training: 58 (N:22, D:36), Testing: 15 (N:6, D:9)
# around 20% testing data
label_training = np.empty((58, 2))
label_training[0:22, :] = [1., 0.]
label_training[22:58, :] = [0., 1.]

label_testing = np.empty((15, 2))
label_testing[0:6, :] = [1., 0.]
label_testing[6:15, :] = [0., 1.]


# model
X = tf.placeholder(tf.float32, [None, 120, 256, 1])
Y = tf.placeholder(tf.float32, [None, 2])
is_training = tf.placeholder(tf.bool)

# L1 Conv shape=(?, 120, 256, 100)
#    Pool     ->(?, 60, 128, 100)
# layer1. n of filter: 100, filter size: 3x3,
L1 = tf.layers.conv2d(X, 16, [3, 3], activation=tf.nn.relu)
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
L1 = tf.layers.dropout(L1, 0.7, is_training)

# L2 Conv shape=(?, 60, 128, 100)
#    Pool     ->(?, 30, 64, 100)
L2 = tf.layers.conv2d(L1, 32, [2, 2])
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
L2 = tf.layers.dropout(L2, 0.7, is_training)

L3 = tf.contrib.layers.flatten(L2)
L3 = tf.layers.dense(L3, 32)
L3 = tf.layers.dropout(L3, 0.5, is_training)

model = tf.layers.dense(L3, 2, activation=None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


# training
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)

batch_size = 58
mean_acc = np.empty([10])

for run in range(np.size(mean_acc)):
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    X_training, X_testing = dt.separate_data_random(X_Normal, X_Dementia, 6, 9)
    X_training = np.expand_dims(X_training, axis=4)
    X_testing = np.expand_dims(X_testing, axis=4)

    for epoch in range(50):

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: X_training,
                                          Y: label_training,
                                          is_training: True})

        # print('Epoch:', '%04d' % (epoch + 1),
        #       'Avg. cost =', '{:.6f}'.format(cost_val))

    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    mean_acc[run] = sess.run(accuracy,
                   feed_dict={X: X_testing,
                              Y: label_testing,
                              is_training: False})
    print("%.2f" %mean_acc[run])
    # print(sess.run(tf.argmax(model, 1),
    #                feed_dict={X: X_testing,
    #                           Y: label_testing,
    #                           is_training: False}))


result = np.sum(mean_acc, 0)/np.size(mean_acc)*100
print("final result: %.2f %%" %result)
