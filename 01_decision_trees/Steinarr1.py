# Author: Steinarr Hrafn
# Date:
# Project: 
# Acknowledgements: 
#


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    targets = np.array(targets)  # just making sure

    ret = list()  # return variable
    N = targets.shape[0]  # number of targets
    if N == 0:
        return 0
    for c in classes:
        ret.append(np.count_nonzero(targets == c)/N)
    return np.array(ret)


def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    split = features[:, split_feature_index] < theta
    features_1 = features[split]
    targets_1 = targets[split]

    not_split = np.logical_not(split)
    features_2 = features[not_split]
    targets_2 = targets[not_split]

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    return 0.5*(1-np.sum(np.square(prior(targets, classes))))


def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = t1.shape[0] + t2.shape[0]
    return (t1.shape[0]*g1 + t2.shape[0]*g2)/n


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''

    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)
    return weighted_impurity(t_1, t_2, classes)

def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        fts_i = features[:, i]  # relevant features
        thetas = np.linspace(np.min(fts_i), np.max(fts_i), num_tries+2)[1:-1]
        # iterate thresholds
        for theta in thetas:
            gini = total_gini_impurity(features, targets, classes, i, theta)

            if gini < best_gini:
                best_gini, best_dim, best_theta = gini, i, theta

    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        return self.tree.score(self.test_features, self.test_targets)

    def plot(self):
        plot_tree(self.tree, fontsize=12)
        plt.show()

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.
        accs = []
        n = self.train_targets.shape[0]
        for i in range(1, n+1):
            t = DecisionTreeClassifier()
            t.fit(self.train_features[:i], self.train_targets[:i])
            accs.append(t.score(self.test_features, self.test_targets))

        plt.plot(accs)
        plt.show()

    def guess(self):
        return self.tree.predict(self.test_features)

    def confusion_matrix(self):

        predictions = self.tree.predict(self.test_features)
        actual = self.test_targets
        matrix = np.zeros((len(self.classes), len(self.classes)), int)
        for i, a in enumerate(self.classes):
            for j, p in enumerate(self.classes):
                matrix[j, i] = np.count_nonzero(predictions[np.where(actual == a)] == p)

        return matrix



if __name__ == "__main__":


    print(f"{'-'*20}\n\t Part 1.1 \n")
    print(f"{prior([0, 0, 1], [0, 1])=}")
    print(f"{prior([0, 2, 3, 3], [0, 1, 2, 3])}=")

    print(f"{'-' * 20}\n\t Part 1.2 \n")
    features, targets, classes = load_iris()
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)

    print(f"{f_1.shape=}")
    print(f"f{f_2.shape=}")


    print(f"{'-' * 20}\n\t Part 1.3 \n")
    print(f"{gini_impurity(t_1, classes)=}")
    print(f"{gini_impurity(t_2, classes)=}")


    print(f"{'-' * 20}\n\t Part 1.4 \n")
    print(f"{weighted_impurity(t_1, t_2, classes)=}")


    print(f"{'-' * 20}\n\t Part 1.5 \n")
    print(f"{total_gini_impurity(features, targets, classes, 2, 4.65)=}")


    print(f"{'-' * 20}\n\t Part 1.6 \n")
    print(f"{brute_best_split(features, targets, classes, 30)=}")

    print(f"{'-' * 20}\n\t Part 2 \n")
    trainer = IrisTreeTrainer(features,targets, classes)
    trainer.train()
    trainer.accuracy()
    print(f"{trainer.accuracy()=}, {trainer.tree.score(trainer.test_features, trainer.test_targets)}")
    # trainer.plot()
    print(f"{trainer.confusion_matrix()=}")
    trainer.plot_progress()

    # print(f"{'-' * 20}\n\t Part 1.4 \n")
    # print(f" = {}")1.4