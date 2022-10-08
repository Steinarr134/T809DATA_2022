import numpy as np


def split_train_test(
    features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.8
):
    '''
    Shuffle the features and targets in unison and return
    two tuples of datasets, first being the training set,
    where the number of items in the training set is according
    to the given train_ratio
    '''
    p = np.random.permutation(features.shape[0])
    features = features[p]
    targets = targets[p]

    split_index = int(features.shape[0] * train_ratio)

    train_features, train_targets = features[0:split_index, :],\
        targets[0:split_index]
    test_features, test_targets = features[split_index:-1, :],\
        targets[split_index: -1]

    return (train_features, train_targets), (test_features, test_targets)
def get_weather_data():
    temps = []
    pressures = []
    suns = []
    winds = []
    month = []
    with open("08_SVM/medalvedur_rvk.txt", 'r') as f:
        for line in f.readlines():
            stuff = line.split("\t")
            temps.append(stuff[3])
            pressures.append(stuff[14])
            suns.append(stuff[16])
            winds.append(stuff[17])
            month.append(stuff[2])
    return split_train_test(np.vstack([temps, pressures, suns, winds]).T, np.array(month).T)
"""
Mánaðarmeðaltöl fyrir stöð 1 - Reykjavík
0       1   2     3    4       5       6      7       8     9       10   11  12  13      14  15 16   17 
stöð	 ár	mán	  t	  tx	  txx	txxD1	  tn	 tnn	tnnD1	 rh	 r	 rx	rxD1	 p	 n	sun	 f
1	1949	1	 -2.7	  0.5	  6.6	9	 -6.8	-15.2	12	 81.0	 68.8	  9.6	25	 996.7	6.0	18.1	 8.5
1	1949	2	  0.0	  2.5	  7.8	6	 -2.5	 -7.5	27	 87.0	 80.0	
"""
if __name__ == "__main__":
    (X_train, t_train), (X_test, t_test) = get_weather_data()
    # print(X.shape, t.shape)