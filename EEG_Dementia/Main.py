import tensorflow as tf
import numpy as np
from EEG_Dementia.DataTransform import create_data
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
X_Normal = np.empty((76, 120, 256))    # 76 subjects, 60 window * 2 channel, 256 -> 1sec
X_Dementia = np.empty((90, 120, 256))   # 90 subjects, 60 window * 2 channel, 256 -> 1sec
X_training = 0
X_testing = 0
label_training = 0
label_testing = 0


# check if data file is exist
if (os.path.isfile('Data/Data_Normal.npy')) and (os.path.isfile('Data/Data_Dementia.npy')):
    X_Normal = np.load('Data/Data_Normal.npy')
    X_Dementia = np.load('Data/Data_Dementia.npy')
else:
    create_data()
    X_Normal = np.load('Data/Data_Normal.npy')
    X_Dementia = np.load('Data/Data_Dementia.npy')



# model
X = tf.placeholder(tf.float32, [None, 120, 256, 1])
Y = tf.placeholder(tf.float32, [None, 1])
is_training = tf.placeholder(tf.bool)


