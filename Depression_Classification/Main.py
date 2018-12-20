import tensorflow as tf
import numpy as np
import Depression_Classification.DataTransform as dt


# # load mat data & save
# dt.load_mat_data_save()

# load data
X_training = np.load('Data/X_training_Normalised_balanced.npy')
X_testing = np.load('Data/X_testing_Normalised.npy')
labelSet_training = np.load('Data/labelSet_training_balanced.npy')
labelSet_testing = np.load('Data/labelSet_testing.npy')
X_training = np.expand_dims(X_training, axis=4)
X_testing = np.expand_dims(X_testing, axis=4)
label_training = dt.one_hot(labelSet_training[:, 1])
label_testing = dt.one_hot(labelSet_testing[:, 1])

print('data loaded')
print(np.shape(X_training), np.shape(label_training))
print(np.shape(X_testing), np.shape(label_testing))


global_step = tf.Variable(0, trainable=False, name='global_step')

# model
X = tf.placeholder(tf.float32, [None, 300, 79, 1])
Y = tf.placeholder(tf.float32, [None, 2])
is_training = tf.placeholder(tf.bool)

# L1 Conv shape=(?, 300, 79, 32)
#    Pool     ->(?, 150, 39, 32)
# layer1. n of filter: 32, filter size: 3x3,
# L1 = tf.layers.conv2d(X, 16, [3, 3], padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=1), bias_initializer=tf.random_uniform_initializer)
L1 = tf.layers.conv2d(X, 32, [3, 3], padding='same', activation=tf.nn.selu, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
L1 = tf.layers.max_pooling2d(L1, [3, 3], [3, 3])
L1 = tf.layers.batch_normalization(L1, training=is_training)
L1 = tf.layers.dropout(L1, 0.5, is_training)
print(np.shape(L1))

# L2 Conv shape=(?, 150, 39, 64)
#    Pool     ->(?, 75, 19, 64)
L2 = tf.layers.conv2d(L1, 64, [2, 2], padding='same', activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
L2 = tf.layers.batch_normalization(L2, training=is_training)
L2 = tf.layers.dropout(L2, 0.5, is_training)
print(np.shape(L2))

# L3 Conv shape=(?, 75, 19, 16)
#    Pool     ->(?, 25, 6, 32)
L3 = tf.layers.conv2d(L2, 128, [3, 2], padding='same', kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
L3 = tf.layers.max_pooling2d(L3, [3, 2], [3, 2])
L3 = tf.layers.batch_normalization(L3, training=is_training)
L3 = tf.layers.dropout(L3, 0.5, is_training)
print(np.shape(L3))

# # L3 Conv shape=(?, 75, 19, 16)
# #    Pool     ->(?, 25, 6, 32)
# L4 = tf.layers.conv2d(L3, 32, [2, 3], padding='same', kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
# L4 = tf.layers.max_pooling2d(L4, [2, 3], [2, 3])
# L4 = tf.layers.batch_normalization(L4, training=is_training)
# L4 = tf.layers.dropout(L4, 0.5, is_training)
# print(np.shape(L4))

L5 = tf.contrib.layers.flatten(L3)
L5 = tf.layers.dense(L5, 79, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
L5 = tf.layers.batch_normalization(L5, training=is_training)
L5 = tf.layers.dropout(L5, 0.5, is_training)
print(np.shape(L5))

model = tf.layers.dense(L5, 2, activation=None)
print(np.shape(model))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost, global_step=global_step)


sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')

if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())


# training
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
        print(i, cost_val)

    print('Epoch:', '%04d' % (epoch + 1),
              'Avg. cost =', '{:.4f}'.format(total_cost / total_batch))

    saver.save(sess, './model/dnn.ckpt', global_step=global_step)


# # testing
# prediction = tf.argmax(model, 1)
# target = tf.argmax(Y, 1)
#
# is_correct = tf.equal(prediction, target)
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
# pred1 = sess.run(prediction, feed_dict={X: X_testing[0:1000, :, :, :], Y: label_testing[0:1000, :], is_training: False})
# print('done')
# # print('정확도 1: %.2f' % sess.run(accuracy * 100, feed_dict={X: X_testing[0:1000, :, :, :], Y: label_testing[0:1000, :], is_training: False}))
# # print('정확도 2: %.2f' % sess.run(accuracy * 100, feed_dict={X: X_testing[1000:2000, :, :, :], Y: label_testing[1000:2000, :], is_training: False}))
# # print('정확도 3: %.2f' % sess.run(accuracy * 100, feed_dict={X: X_testing[2000:3000, :, :, :], Y: label_testing[2000:3000, :], is_training: False}))
# # print('정확도 4: %.2f' % sess.run(accuracy * 100, feed_dict={X: X_testing[3000:4000, :, :, :], Y: label_testing[3000:4000, :], is_training: False}))
# # print('정확도 5: %.2f' % sess.run(accuracy * 100, feed_dict={X: X_testing[4000:5000, :, :, :], Y: label_testing[4000:5000, :], is_training: False}))
# # print('정확도 6: %.2f' % sess.run(accuracy * 100, feed_dict={X: X_testing[5000:6000, :, :, :], Y: label_testing[5000:6000, :], is_training: False}))
# # print('정확도 7: %.2f' % sess.run(accuracy * 100, feed_dict={X: X_testing[6000:7000, :, :, :], Y: label_testing[6000:7000, :], is_training: False}))


# # testing (majority vote)
# prediction = tf.argmax(model, 1)
# pred = sess.run(prediction, feed_dict={X: X_testing, Y: label_testing, is_training: False})
#
# for i in range(np.size(labelSet_testing, 0)):
#     print(i)
#
# print(type(pred))
# print(np.shape(pred))
#
# print('done')