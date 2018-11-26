from scipy.io import loadmat
import numpy as np
import h5py


# Use matlab code instead (faster)
# save X_training, X_testing, labelSet_training, labelSet_testing
# X_training (N, 300, 79): 300 indicate 3s features
# X_testing (M, 300, 79)
# labelSet_training (N, 3): N * [id, PHQ8 binary, PHQ8 score]
# labelSet_testing (M,3)
def create_data():

    X_training = np.empty([0, 300, 79])
    train_set = loadmat('Data/train_set.mat')['train_set']
    # print(np.size(train_set,0))

    for i in range(0, np.size(train_set, 0)):

        # print(train_set[i, 0])
        tmpData = loadmat('Data/Features/'+str(train_set[i, 0])+'_P_features.mat')['features']

        for j in range(0, int(np.size(tmpData, 0)/300)):
            tmp = np.expand_dims(tmpData[j*300:j*300+300, :], axis=0)
            X_training = np.append(X_training, tmp, axis=0)


        print(train_set[i, 0])

    print("FINISHED!")
    np.save('Data/X_training', X_training)


def load_mat_data_save():

    # X_training.mat: mat version 7.3 as the size is too big - it requires different mat loading code
    f = h5py.File('Data/Data_mat/X_training.mat', 'r')
    data = f.get('X_training')
    data = np.array(data)
    X_training = np.swapaxes(data,0,2)
    print(np.shape(X_training))

    X_testing = loadmat('Data/Data_mat/X_testing.mat')['X_testing']
    print(np.shape(X_testing))

    labelSet_training = loadmat('Data/Data_mat/labelSet_training.mat')['labelSet_training']
    labelSet_testing = loadmat('Data/Data_mat/labelSet_testing.mat')['labelSet_testing']
    print(np.shape(labelSet_training))
    print(np.shape(labelSet_testing))

    np.save('Data/X_training', X_training)
    np.save('Data/X_testing', X_testing)
    np.save('Data/labelSet_training', labelSet_training)
    np.save('Data/labelSet_testing', labelSet_testing)


def one_hot(indices):
    oh = np.zeros((indices.size, indices.max()+1))
    oh[np.arange(indices.size), indices] = 1

    return oh
