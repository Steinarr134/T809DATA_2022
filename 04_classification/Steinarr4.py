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
    posterior = maximum_likelihood(train_features, train_targets, test_features, classes)

    return posterior*np.array(priors)

def confusion_matrix(classes, predictions, actual):
    matrix = np.zeros((len(classes), len(classes)), int)
    for i, a in enumerate(classes):
        for j, p in enumerate(classes):
            matrix[j, i] = np.count_nonzero(predictions[np.where(actual == a)] == p)

    return matrix


if __name__ == '__main__':
    np.random.seed(1234)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) \
        = split_train_test(features, targets, train_ratio=0.6)

    # set to True for Indepenent section
    if False:
        np.random.seed(1234)
        n = 1000
        features = np.random.randint(0, 2, (n, 10))
        targets = np.random.randint(0, 4, n)
        classes = np.array([0, 1, 2, 3])

        # remove two thirds of datapoints when the die threw 0 or 1
        # thus making the die unfair
        some_positions = np.where(targets <= 1)[0]
        positions2remove = some_positions[some_positions.shape[0] // 4:]
        targets = np.delete(targets, positions2remove)
        features = np.delete(features, positions2remove, axis=0)

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

    print(f"\n\n{'-' * 20}\n\t Part 2\n")
    post_likelihoods = maximum_aposteriori(train_features, train_targets, test_features, classes)
    print(predict(likelihoods) == test_targets)
    print(confusion_matrix(classes, test_targets, predict(likelihoods)))
    print(confusion_matrix(classes, test_targets, predict(post_likelihoods)))


    def format_matrix(matrix, environment="bmatrix", formatter=str):
        """Format a matrix using LaTeX syntax"""

        if not isinstance(matrix, np.ndarray):
            try:
                matrix = np.array(matrix)
            except Exception:
                raise TypeError("Could not convert to Numpy array")

        if len(shape := matrix.shape) == 1:
            matrix = matrix.reshape(1, shape[0])
        elif len(shape) > 2:
            raise ValueError("Array must be 2 dimensional")

        body_lines = [" & ".join(map(formatter, row)) for row in matrix]

        body = "\\\\\n".join(body_lines)
        return f"""\\begin{{{environment}}}
    {body}
    \\end{{{environment}}}"""

    print(format_matrix(confusion_matrix(classes, test_targets, predict(likelihoods))))
    print(format_matrix(confusion_matrix(classes, test_targets, predict(post_likelihoods))))
    print(np.count_nonzero(test_targets == predict(likelihoods))/test_targets.shape[0])
    print(np.count_nonzero(test_targets == predict(post_likelihoods))/test_targets.shape[0])

    # print(f"\n\n{'-' * 20}\n\t Independent\n")

    # # remove some train features from class 1
    # if True:
    #     c1_positions = np.where(train_targets == 1)[0]
    #     positions2remove = c1_positions[c1_positions.shape[0] // 3:]
    #     train_targets = np.delete(train_targets, positions2remove)
    #     train_features = np.delete(train_features, positions2remove, axis=0)
    #     c1_positions = np.where(test_targets == 1)[0]
    #     positions2remove = c1_positions[c1_positions.shape[0] // 3:]
    #     test_targets = np.delete(test_targets, positions2remove)
    #     test_features = np.delete(test_features, positions2remove, axis=0)
    #     print(train_targets)





