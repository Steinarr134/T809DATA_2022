# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


from tools import plot_svm_margin, load_cancer
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt


def _plot_linear_kernel(show=True):
    X, t = make_blobs(40, centers=2)
    
    svc = svm.SVC(C=1000, kernel="linear")
    svc.fit(X, t)
    print(" number of support vectors for each class:", svc.n_support_)
    print("thus the shape of decision boundary is a line")

    plot_svm_margin(svc, X, t)
    if show:
        plt.show()

if __name__ == "_Q_main__":
    _plot_linear_kernel(show=False)
    plt.gcf().set_figheight(6)
    plt.gcf().set_figwidth(9)
    plt.savefig("08_SVM/1_1_1.png")
    plt.show()


def _subplot_svm_margin(
    svc,
    X: np.ndarray,
    t: np.ndarray,
    num_plots: int,
    index: int,
    title=None
):
    '''
    Plots the decision boundary and decision margins
    for a dataset of features X and labels t and a support
    vector machine svc.

    Input arguments:
    * svc: An instance of sklearn.svm.SVC: a C-support Vector
    classification model
    * X: [N x f] array of features
    * t: [N] array of target labels
    '''
    # similar to tools.plot_svm_margin but added num_plots and
    # index where num_plots should be the total number of plots
    # and index is the index of the current plot being generated
    plt.subplot(1, num_plots, index)
    if title is not None:
        plt.gca().title.set_text(title)
    plot_svm_margin(svc, X, t)


def _compare_gamma(show=True):
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)

    clf = svm.SVC(C=1000, kernel="rbf")
    clf.fit(X,t)
    print(f"For default gamma, number of support vectors is: {clf.n_support_}")
    _subplot_svm_margin(clf, X, t, 3, 1, title="gamma=default")

    clf = svm.SVC(C=1000, kernel="rbf", gamma=0.2)
    clf.fit(X, t)
    print(f"For gamma=0.2, number of support vectors is: {clf.n_support_}")
    _subplot_svm_margin(clf, X, t, 3, 2, title="gamma=0.2")


    clf = svm.SVC(C=1000, kernel="rbf", gamma=2, )
    clf.fit(X, t)
    print(f"For gamma=2, number of support vectors is: {clf.n_support_}")
    _subplot_svm_margin(clf, X, t, 3, 3, title="gamma=2")

    if show:
        plt.show()

if __name__ == "_Q_main__":
    _compare_gamma(show=False)
    plt.gcf().set_figheight(6)
    plt.gcf().set_figwidth(9)
    plt.savefig("08_SVM/1_3_1.png")
    plt.show()


def _compare_C(show=True):
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)

    Cs = [1000, 0.5, 0.3, 0.05, 0.0001]
    for i, C in enumerate(Cs):
        clf = svm.SVC(C=C, kernel="linear")
        clf.fit(X, t)
        print(f"For C={C}, number of support vectors is: {clf.n_support_}")
        _subplot_svm_margin(clf, X, t, len(Cs), i+1, title=f"{C=}")
    if show:
        plt.show()

if __name__ == "__main__":
    _compare_C(False)
    plt.gcf().set_figheight(6)
    plt.gcf().set_figwidth(9)
    plt.savefig("08_SVM/1_5_1.png")
    plt.show()

        
def train_test_SVM(
    svc,
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
):
    '''
    Train a configured SVM on <X_train> and <t_train>
    and then measure accuracy, precision and recall on
    the test set

    This function should return (accuracy, precision, recall)
    '''
    svc.fit(X_train, t_train)
    y = svc.predict(X_test)
    return accuracy_score(t_test, y), precision_score(t_test, y), recall_score(t_test, y)

if __name__ == "_Q_main__":
    (X_train, t_train), (X_test, t_test) = load_cancer()

    kernels = ["linear", "rbf","poly"]
    for kernel in kernels:
        svc = svm.SVC(C=1000, kernel=kernel)
        print(f"Using {kernel=}, gives: {train_test_SVM(svc, X_train, t_train, X_test, t_test)}")
        
""" Independent"""
if __name__ == "__main__":


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
        
    (X_train, t_train), (X_test, t_test) = get_weather_data()
    # kernels = ["linear", "rbf", "poly"]
    # for kernel in kernels:
    #     svc = svm.SVC(C=1000, kernel=kernel,)
    #     print(f"Using {kernel=}, gives: {train_test_SVM(svc, X_train, t_train, X_test, t_test)}")

    svc = svm.SVC(C=1000, kernel='linear')
    svc.fit(X_train, t_train),
    y = svc.predict(X_test)
    confuse = confusion_matrix(t_test, y)


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

        classes = ("January",
                   "February",
                   "March",
                   "April",
                   "May",
                   "June",
                   "July",
                   "August",
                   "September",
                   "October",
                   "November",
                   "December")
        body_lines = [classes[i] + " & " +
                    " & ".join(map(formatter, row)) for i, row in enumerate(matrix)]

        body = "\\\\\n".join(body_lines)
        return f"""\\begin{{{environment}}}
        {"(empty) &" + " & ".join(classes)}\\\\
        {body}
        \\end{{{environment}}}"""

    print(format_matrix(confuse))
    print(t_test.shape)