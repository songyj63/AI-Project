import tensorflow as tf
import numpy as np
import EEG_Dementia.DataTransform as dt
import os.path


# FFT data
# Freq series data
# 0-50 Hz FFT 2 channel data
# each participant: [3000 2]
# [0-50 freq info, 2ch]


# load data
# X_Normal - 28 subjects, 3000 (fft freq 0-50 Hz), 2 channel
# X_Dementia - 45 subjects, 3000 (fft freq 0-50 Hz), 2 channel
# check if data file is exist and load
if (os.path.isfile('Data/Data_Normal_FFT.npy')) and (os.path.isfile('Data/Data_Dementia_FFT.npy')):
    X_Normal = np.load('Data/Data_Normal_FFT.npy')
    X_Dementia = np.load('Data/Data_Dementia_FFT.npy')
else:
    print("create data")
    dt.create_data_fft(28, 45)  # X_Normal size, X_Dementia size
    X_Normal = np.load('Data/Data_Normal_FFT.npy')
    X_Dementia = np.load('Data/Data_Dementia_FFT.npy')

print(np.shape(X_Normal))
print(np.shape(X_Dementia))