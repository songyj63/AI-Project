import numpy as np


def create_data():

    # Data description
    # 2 channel EEG data. 1min, 256 s/s -> 15360 * 2ch
    # reshape to [60*2, 256]
    # ch1 0-1s data [n, 256]
    # ch2 0-1s data
    # ch1 1-2s data
    # ch2 1-2s data

    X_Normal = np.empty((76, 120, 256))  # 76 subjects, 60 window * 2 channel, 256 -> 1sec
    X_Dementia = np.empty((90, 120, 256))  # 90 subjects, 60 window * 2 channel, 256 -> 1sec

    for i in range(1, 77):
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

    for i in range(1, 91):
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



