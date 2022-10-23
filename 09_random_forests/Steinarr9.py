# Author:
# Date:
# Project:
# Acknowledgements:
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score)

from collections import OrderedDict

class CancerClassifier:
    '''
    A general class to try out different sklearn classifiers
    on the cancer dataset
    '''

    def __init__(self, classifier, train_ratio: float = 0.7):
        self.classifier = classifier
        cancer = load_breast_cancer()
        self.X = cancer.data  # all feature vectors
        self.t = cancer.target  # all corresponding labels
        self.X_train, self.X_test, self.t_train, self.t_test =\
            train_test_split(
                cancer.data, cancer.target,
                test_size=1-train_ratio, random_state=109)

        # Fit the classifier to the training data here
        self.classifier.fit(self.X_train, self.t_train, )
        self._test_predictions = None
        # self.classes = [0, 1]

    @property
    def test_predictions(self):
        if self._test_predictions is None:
            self._test_predictions = self.classifier.predict(self.X_test)
        return self._test_predictions
    

    def confusion_matrix(self) -> np.ndarray:
        '''Returns the confusion matrix on the test data
        '''
        return confusion_matrix(self.t_test, self.test_predictions)
        # matrix = np.zeros((len(self.classes), len(self.classes)), int)
        # for i, a in enumerate(self.classes):
        #     for j, p in enumerate(self.classes):
        #         matrix[j, i] = np.count_nonzero(
        #             self.test_predictions[np.where(self.t_train == a)] == p)
        # return matrix

    def accuracy(self) -> float:
        '''Returns the accuracy on the test data
        '''
        return accuracy_score(self.t_test, self.test_predictions)

    def precision(self) -> float:
        '''Returns the precision on the test data
        '''
        return precision_score(self.t_test, self.test_predictions)

    def recall(self) -> float:
        '''Returns the recall on the test data
        '''
        return recall_score(self.t_test, self.test_predictions)

    def cross_validation_accuracy(self) -> float:
        '''Returns the average 10-fold cross validation
        accuracy on the entire dataset.
        '''
        return np.average(cross_val_score(self.classifier, self.X, self.t, cv=10))

    def feature_importance(self, save_name=None) -> list:
        '''
        Draw and show a barplot of feature importances
        for the current classifier and return a list of
        indices, sorted by feature importance (high to low).
        '''
        ret = np.flip(np.argsort(self.classifier.feature_importances_))
        plt.bar(np.array(range(len(self.classifier.feature_importances_))), self.classifier.feature_importances_)
        plt.xlabel("Feature index")
        plt.ylabel("Feature importance")
        if save_name is not None:
            plt.savefig(save_name)
        print(self.classifier.feature_importances_)
        print(ret)
        plt.show()


"""
Section 1
"""
if __name__ == "__main__":
    classifier_type = DecisionTreeClassifier()
    cc = CancerClassifier(classifier_type)
    print("SEction 1")
    print(f"{cc.confusion_matrix()=}")
    print(f"{cc.cross_validation_accuracy()=}")
    print(f"{cc.accuracy()=}")
    print(f"{cc.precision()=}")
    print(f"{cc.recall()=}")


"""Section 2.1"""
if __name__ == "__main__":
    classifier_type = RandomForestClassifier(n_estimators=120, max_features=4)
    cc = CancerClassifier(classifier_type)
    print("SEction 2")
    print(f"{cc.confusion_matrix()=}")
    print(f"{cc.cross_validation_accuracy()=}")
    print(f"{cc.accuracy()=}")
    print(f"{cc.precision()=}")
    print(f"{cc.recall()=}")
    print(f"""The confusion matrix was:

          \\begin[center]
          CM= \\begin[bmatrix]
          {cc.confusion_matrix()[0, 0]} & {cc.confusion_matrix()[0, 1]} 
          {cc.confusion_matrix()[1, 0]}  & {cc.confusion_matrix()[1, 1]} 
          \end[bmatrix]
          \end[center]


          Accuracy was {cc.accuracy():.2%}\% , Precision was {cc.precision():.2%}\% , recall was {cc.recall():.2%}\% and cross validation accuracy was {cc.cross_validation_accuracy():.2%}\% . """.replace('[', '{').replace(']', '}'))
    # cc.feature_importance("09_random_forests/2_2_1.png")
    # quit()

# """
# Section 2
# """
# if __name__ == "__main__":
    # cancer = load_breast_cancer()
    # X = cancer.data  # all feature vectors
    # t = cancer.target  # all corresponding labels
    # X_train, X_test, t_train, t_test =\
    #     train_test_split(
    #         cancer.data, cancer.target,
    #         test_size=0.3, random_state=109)
    # rf = RandomForestClassifier()
    # rf.fit(X_train, t_train)
    # predictions = rf.predict(X_test)

    # print("\n\n Section 2")
    # print(f"{confusion_matrix(t_test, predictions)=}")
    # print(f"{accuracy_score(t_test, predictions)=}")
    # print(f"{precision_score(t_test, predictions)=}")
    # print(f"{recall_score(t_test, predictions)=}")
    # print(f"{np.average(cross_val_score(rf, X, t, cv=10))}")

    # best_score = [0, 0, 0]
    # best_combo = None
    # for ne in range(50, 200, 10):
    #     for mf in range(1, 10):
    #         rf = RandomForestClassifier(n_estimators=ne, max_features=mf)
    #         rf.fit(X_train, t_train)
    #         predictions = rf.predict(X_test)
    #         acc = accuracy_score(t_test, predictions)
    #         prec = precision_score(t_test, predictions)
    #         rec = recall_score(t_test, predictions)
    #         if rec >= best_score[0]:
    #             if acc >= best_score[1]:
    #                 if prec >= best_score[2]:
    #                     best_score = [rec, acc, prec]
    #                     best_combo = (ne, mf)
    # print(f"{best_combo=}, {best_score=}")
    # rf = RandomForestClassifier(n_estimators=best_combo[0], max_features=best_combo[1])
    # rf = RandomForestClassifier(n_estimators=120, max_features=4)

    # print(f"{np.average(cross_val_score(rf, X, t, cv=10))}")
    # rf.fit(X_train, t_train)
    # predictions = rf.predict(X_test)

def _plot_oob_error(save_as=None, smooth_fun=None):
    RANDOM_STATE = 1337
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 300
    cancer = load_breast_cancer()
    X = cancer.data  # all feature vectors
    t = cancer.target  # all corresponding labels

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X, t)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        if smooth_fun is not None:
            ys = smooth_fun(ys)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()

if __name__ == "__main__":
    from scipy.ndimage import uniform_filter1d
    smoother = lambda y: uniform_filter1d(y, 5)
#     _plot_oob_error("09_random_forests/2_4_1_smoothed.png", smooth_fun=smoother)

def _plot_extreme_oob_error(save_as=None, smooth_fun=None):
    RANDOM_STATE = 1337
    ensemble_clfs = [
        ("ExtraTreesClassifier, max_features='sqrt'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                bootstrap=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features='log2'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                bootstrap=True,
                max_features='log2',
                oob_score=True,
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features=None",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                bootstrap=True,
                max_features=None,
                oob_score=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 300
    cancer = load_breast_cancer()
    X = cancer.data  # all feature vectors
    t = cancer.target  # all corresponding labels

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X, t)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        if smooth_fun is not None:
            ys = smooth_fun(ys)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()


"""Section 3"""
if __name__ == "__main__":
    classifier_type = ExtraTreesClassifier()
    cc = CancerClassifier(classifier_type)
    print("SEction 2")
    print(f"{cc.confusion_matrix()=}")
    print(f"{cc.cross_validation_accuracy()=}")
    print(f"{cc.accuracy()=}")
    print(f"{cc.precision()=}")
    print(f"{cc.recall()=}")
    print(f"""The confusion matrix was:

          \\begin[center]
          CM= \\begin[bmatrix]
          {cc.confusion_matrix()[0, 0]} & {cc.confusion_matrix()[0, 1]} 
          {cc.confusion_matrix()[1, 0]}  & {cc.confusion_matrix()[1, 1]} 
          \end[bmatrix]
          \end[center]


          Accuracy was {cc.accuracy():.2%} , Precision was {cc.precision():.2%} , recall was {cc.recall():.2%} and cross validation accuracy was {cc.cross_validation_accuracy():.2%} . """.replace('[', '{').replace(']', '}').replace('%', '\%'))
    # cc.feature_importance("09_random_forests/3_1_1.png")

    _plot_extreme_oob_error("09_random_forests/3_2_1.png", smooth_fun=smoother)
    