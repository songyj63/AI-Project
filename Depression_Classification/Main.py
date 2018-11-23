import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import h5py


# # load mat data & save
# # X_training.mat: mat version 7.3 as the size is too big - it requires different mat loading code
# f = h5py.File('Data/Data_mat/X_training.mat', 'r')
# data = f.get('X_training')
# data = np.array(data)
# X_training = np.swapaxes(data,0,2)
# print(np.shape(X_training))
#
# X_testing = loadmat('Data/Data_mat/X_testing.mat')['X_testing']
# print(np.shape(X_testing))
#
# labelSet_training = loadmat('Data/Data_mat/labelSet_training.mat')['labelSet_training']
# labelSet_testing = loadmat('Data/Data_mat/labelSet_testing.mat')['labelSet_testing']
# print(np.shape(labelSet_training))
# print(np.shape(labelSet_testing))
#
# np.save('Data/X_training', X_training)
# np.save('Data/X_testing', X_testing)
# np.save('Data/labelSet_training', labelSet_training)
# np.save('Data/labelSet_testing', labelSet_testing)


# load data
X_training = np.load('Data/X_training.npy')
X_testing = np.load('Data/X_testing.npy')
labelSet_training = np.load('Data/labelSet_training.npy')
labelSet_testing = np.load('Data/labelSet_testing.npy')
print('data loaded')


