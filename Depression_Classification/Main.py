import tensorflow as tf
import numpy as np
import Depression_Classification.DataTransform as dt


# # load mat data & save
# dt.load_mat_data_save()


# load data
X_training = np.load('Data/X_training.npy')
X_testing = np.load('Data/X_testing.npy')
labelSet_training = np.load('Data/labelSet_training.npy')
labelSet_testing = np.load('Data/labelSet_testing.npy')
X_training = np.expand_dims(X_training, axis=4)
X_testing = np.expand_dims(X_testing, axis=4)
label_training = dt.one_hot(labelSet_training[:, 1]);
label_testing = dt.one_hot(labelSet_testing[:, 1]);
print('data loaded')
print(np.shape(X_training), np.shape(X_testing))


# model
X = tf.placeholder(tf.float32, [None, 300, 79, 1])
Y = tf.placeholder(tf.float32, [None, 2])
is_training = tf.placeholder(tf.bool)

# L1 Conv shape=(?, 300, 79, 32)
#    Pool     ->(?, 150, 39, 32)
# layer1. n of filter: 32, filter size: 3x3,
L1 = tf.layers.conv2d(X, 64, [3, 3], padding='same', activation=tf.nn.relu)
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
L1 = tf.layers.dropout(L1, 0.7, is_training)
print(np.shape(L1))

# L2 Conv shape=(?, 150, 39, 128)
#    Pool     ->(?, 75, 19, 128)
L2 = tf.layers.conv2d(L1, 128, [2, 2], padding='same')
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
L2 = tf.layers.dropout(L2, 0.7, is_training)
print(np.shape(L2))

# L3 Conv shape=(?, 75, 19, 32)
#    Pool     ->(?, 25, 6, 32)
L3 = tf.layers.conv2d(L2, 32, [3, 2], padding='same')
L3 = tf.layers.max_pooling2d(L3, [3, 3], [3, 3])
L3 = tf.layers.dropout(L3, 0.7, is_training)
print(np.shape(L3))

L4 = tf.contrib.layers.flatten(L3)
L4 = tf.layers.dense(L4, 79)
L4 = tf.layers.dropout(L4, 0.5, is_training)
print(np.shape(L4))

model = tf.layers.dense(L4, 2, activation=None)
print(np.shape(model))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


# # training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(np.size(X_training, 0) / batch_size)

for epoch in range(50):
    total_cost = 0
    for i in range(total_batch):
        batch_xs = X_training[batch_size * i:batch_size * (i + 1), :, :, :]
        batch_ys = label_training[batch_size * i:batch_size * (i + 1), :]
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          is_training: True})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
              'Avg. cost =', '{:.4f}'.format(total_cost / total_batch))

# print("%.2f" % mean_acc[run])
# print(sess.run(tf.argmax(model, 1),
#                feed_dict={X: X_testing,
#                           Y: label_testing,
#                           is_training: False}))