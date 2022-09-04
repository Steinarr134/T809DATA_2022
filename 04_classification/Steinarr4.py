# Author: Steinarr Hrafn
# Date: 4-? sept 2022
# Project:
# Acknowledgements:
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(
        features: np.ndarray,
        targets: np.ndarray,
        selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    return np.mean(features[np.where(targets==selected_class)], axis=0)


def covar_of_class(
        features: np.ndarray,
        targets: np.ndarray,
        selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    features = features[np.where(targets==selected_class)]
    return np.cov(features, rowvar=False)


def likelihood_of_class(
        feature: np.ndarray,
        class_mean: np.ndarray,
        class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    return multivariate_normal(mean=class_mean, cov=class_covar).pdf(feature)


def maximum_likelihood(
        train_features: np.ndarray,
        train_targets: np.ndarray,
        test_features: np.ndarray,
        classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))
    likelihoods = []
    for test_point in test_features:
        likelihood = []
        for mean, cov in zip(means, covs):
            likelihood.append(likelihood_of_class(test_point, mean, cov))
        likelihoods.append(likelihood)
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    return np.argmax(likelihoods, axis=1)


def maximum_aposteriori(
        train_features: np.ndarray,
        train_targets: np.ndarray,
        test_features: np.ndarray,
        classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    priors = []
    for c in classes:
        priors.append(np.count_nonzero(train_targets == c) / len(train_targets))
    


if __name__ == '__main__':
    np.random.seed(1234)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) \
        = split_train_test(features, targets, train_ratio=0.6)
    print(f"\n\n{'-' * 20}\n\t Part 1.1 \n")
    print(f"{mean_of_class(train_features, train_targets, 0)=}")


    print(f"\n\n{'-' * 20}\n\t Part 1.2 \n")
    print(f"{ covar_of_class(train_features, train_targets, 0)=}")

    print(f"\n\n{'-' * 20}\n\t Part 1.3 \n")
    class_mean = mean_of_class(train_features, train_targets, 0)
    class_cov = covar_of_class(train_features, train_targets, 0)
    print(f"{likelihood_of_class(test_features[0, :], class_mean, class_cov)=}")

    print(f"\n\n{'-' * 20}\n\t Part 1.4 \n")
    print(f"{maximum_likelihood(train_features, train_targets, test_features, classes)=}")

    print(f"\n\n{'-' * 20}\n\t Part 1.5 \n")
    likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    print(f"{predict(likelihoods)}")

    print(f"\n\n{'-' * 20}\n\t Part 2.1 \n")









