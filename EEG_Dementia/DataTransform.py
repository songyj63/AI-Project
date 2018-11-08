import numpy as np


def create_data_raw(size_n, size_d):

    # Data description
    # 2 channel EEG data. 1min, 256 s/s -> 15360 * 2ch
    # reshape to [60*2, 256]
    # ch1 0-1s data [n, 256]
    # ch2 0-1s data
    # ch1 1-2s data
    # ch2 1-2s data

    X_Normal = np.empty((size_n, 120, 256))  # 28 subjects, 60 window * 2 channel, 256 -> 1sec
    X_Dementia = np.empty((size_d, 120, 256))  # 45 subjects, 60 window * 2 channel, 256 -> 1sec

    for i in range(1, size_n+1):
        fPath = "Data/Normal/N (" + str(i) + ").txt"
        file_obj = open(fPath, "r")
        mat = file_obj.read()
        arr = np.matrix(mat)
        arr = np.reshape(arr, (-1, 3))
        arr = (arr[:, 1:3])
        arr1 = np.reshape(arr[:, 0], (-1, 256))
        arr2 = np.reshape(arr[:, 1], (-1, 256))
        arr = np.append(arr1, arr2, axis=0)
        X_Normal[i - 1, :, :] = arr

    for i in range(1, size_d+1):
        fPath = "Data/Dementia/D (" + str(i) + ").txt"
        file_obj = open(fPath, "r")
        mat = file_obj.read()
        arr = np.matrix(mat)
        arr = np.reshape(arr, (-1, 3))
        arr = (arr[:, 1:3])
        arr1 = np.reshape(arr[:, 0], (-1, 256))
        arr2 = np.reshape(arr[:, 1], (-1, 256))
        arr = np.append(arr1, arr2, axis=0)
        X_Dementia[i - 1, :, :] = arr

    np.save('Data/Data_Normal', X_Normal)
    np.save('Data/Data_Dementia', X_Dementia)


def create_data_fft(size_n, size_d):

    # Freq series data
    # 0-50 Hz FFT 2 channel data
    # each participant: [3000 2]
    # [0-50 freq info, 2ch]

    X_Normal_FFT = np.empty((size_n, 3000, 2))  # 28 subjects, 3000 (fft freq 0-50 Hz), 2 channel
    X_Dementia_FFT = np.empty((size_d, 3000, 2))  # 45 subjects, 3000 (fft freq 0-50 Hz), 2 channel

    for i in range(1, size_n + 1):
        fPath = "Data/Data_FFT/Normal/N (" + str(i) + ")_FFT.csv"
        file_obj = open(fPath, "r")
        mat = file_obj.read()
        arr = np.matrix(mat)
        arr = np.reshape(arr, (-1, 2))
        X_Normal_FFT[i - 1, :, :] = arr

    for i in range(1, size_d + 1):
        fPath = "Data/Data_FFT/Dementia/D (" + str(i) + ")_FFT.csv"
        file_obj = open(fPath, "r")
        mat = file_obj.read()
        arr = np.matrix(mat)
        arr = np.reshape(arr, (-1, 2))
        X_Dementia_FFT[i - 1, :, :] = arr

    np.save('Data/Data_Normal_FFT', X_Normal_FFT)
    np.save('Data/Data_Dementia_FFT', X_Dementia_FFT)


def separate_data_random(x_normal, x_dementia, nTestingSize, dTestingSize):

    nTotalSize = np.size(x_normal, axis=0)
    dTotalSize = np.size(x_dementia, axis=0)

    nR = np.random.choice(range(nTotalSize), nTestingSize, replace=False)
    dR = np.random.choice(range(dTotalSize), dTestingSize, replace=False)

    nTesting = x_normal[nR,:]
    dTesting = x_dementia[dR,:]

    nTraining = np.delete(x_normal, nR, axis=0)
    dTraining = np.delete(x_dementia, dR, axis=0)

    X_training = np.append(nTraining, dTraining, axis=0)
    X_testing = np.append(nTesting, dTesting, axis=0)

    return X_training, X_testing



def separate_data(x_normal, x_dementia, nTestingSize, dTestingSize):

    nTotalSize = np.size(x_normal, axis=0)
    dTotalSize = np.size(x_dementia, axis=0)

    nTraining = x_normal[0:nTotalSize-nTestingSize, :]
    dTraining = x_dementia[0:dTotalSize-dTestingSize, :]

    nTesting = x_normal[nTotalSize-nTestingSize:, :]
    dTesting = x_dementia[dTotalSize-dTestingSize:, :]

    X_training = np.append(nTraining, dTraining, axis=0)
    X_testing = np.append(nTesting, dTesting, axis=0)

    return X_training, X_testing