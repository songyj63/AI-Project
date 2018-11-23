from scipy.io import loadmat
import numpy as np


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