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
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
    index: int
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
    plot_svm_margin(svc, X, t)


def _compare_gamma(show=True):
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)

    clf = svm.SVC(C=1000, kernel="rbf")
    clf.fit(X,t)
    print(f"For default gamma, number of support vectors is: {clf.n_support_}")
    _subplot_svm_margin(clf, X, t, 3, 1)

    clf = svm.SVC(C=1000, kernel="rbf", gamma=0.2)
    clf.fit(X, t)
    print(f"For gamma=0.2, number of support vectors is: {clf.n_support_}")
    _subplot_svm_margin(clf, X, t, 3, 2)


    clf = svm.SVC(C=1000, kernel="rbf", gamma=2)
    clf.fit(X, t)
    print(f"For gamma=2, number of support vectors is: {clf.n_support_}")
    _subplot_svm_margin(clf, X, t, 3, 3)

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
        _subplot_svm_margin(clf, X, t, len(Cs), i+1)
    if show:
        plt.show()

if __name__ == "_Q_main__":
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

if __name__ == "__main__":
    (X_train, t_train), (X_test, t_test) = load_cancer()

    kernels = ["linear", "rbf","poly"]
    for kernel in kernels:
        svc = svm.SVC(C=1000, kernel=kernel)
        print(f"Using {kernel=}, gives: {train_test_SVM(svc, X_train, t_train, X_test, t_test)}")
        

