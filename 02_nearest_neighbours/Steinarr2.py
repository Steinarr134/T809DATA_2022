# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    return np.sqrt(np.sum(np.square(x - y)))


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    distances = euclidian_distances(x, points)
    return np.argsort(distances)[:k]



def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    return classes[np.argmax(np.bincount(targets))]


def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    k_n = k_nearest(x, points, k)
    return vote(point_targets[k_n], classes)


def remove_one(points: np.ndarray, i: int):
    '''
    Removes the i-th from points and returns
    the new array
    '''
    return np.concatenate((points[0:i], points[i+1:]))


def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    predictions = []
    for i in range(points.shape[0]):
        predictions.append(knn(points[i], remove_one(points, i), remove_one(point_targets, i), classes, k))
    return np.array(predictions)


def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    predictions = knn_predict(points, point_targets, classes, k)
    return np.average(point_targets == predictions)


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    predictions = knn_predict(points, point_targets, classes, k)
    matrix = np.zeros((len(classes), len(classes)), int)
    print(point_targets)
    print(predictions)
    for i, a in enumerate(classes):
        for j, p in enumerate(classes):
            matrix[j, i] = np.count_nonzero(predictions[np.where(point_targets == a)] == p)
    return matrix


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    n = points.shape[0]
    best, where = 0, None
    for k in range(1, n):
        k_acc = knn_accuracy(points, point_targets, classes, k)
        if k_acc > best:
            best, where = k_acc, k
    return where


def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    predictions = knn_predict(points, point_targets, classes, k)

    colors = ['yellow', 'purple', 'blue']
    for i in range(points.shape[0]):
        [x, y] = points[i,:2]
        c = "green" if predictions[i] == point_targets[i] else "red"
        plt.scatter(x, y, c=colors[point_targets[i]], edgecolors=c,
                    linewidths=2)
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.show()



def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    if np.any(distances == 0):
        return classes[np.where(distances == 0)[0].flat[0]]
    cs = np.divide(1, distances)
    csum = np.sum(cs)
    weights = np.divide(cs, csum)  # I think dividing by csum does nothing since we take argmax anyway...

    # here is a one-liner that expands weights into the shape of targets, multiplies those two matrixes and
    # then finally sums it in one axis. the end results is a vector of same length as classes.
    votesum = np.sum(np.repeat(weights, targets.shape[1]).reshape(targets.shape)*targets, axis=0)

    return classes[np.argmax(votesum)]

def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    # a copy of classes as a numpy array for indexing
    np_classes = np.array(classes)
    # get the k nearest neighbours, same as in knn
    k_ns = k_nearest(x, points, k)

    # express the targets as a vector of vectors
    targets = np.zeros((k_ns.shape[0], len(classes)))
    for i, point_n in enumerate(k_ns):
        # i is just the row, but the position has to match the position of the class in classes
        # so we have to find the target corrisponding to point_n and find its position in classes
        targets[i, np.argwhere(np_classes==point_targets[point_n]).flat[0]] = 1


    # fin the distances of the points from x
    distances = euclidian_distances(x, points[k_ns])

    return weighted_vote(targets, distances, classes)


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    predictions = []
    for i in range(points.shape[0]):
        predictions.append(wknn(points[i], remove_one(points, i), remove_one(point_targets, i), classes, k))
    return np.array(predictions)


def wknn_accuracy(
        points: np.ndarray,
        point_targets: np.ndarray,
        classes: list,
        k: int
) -> float:
    predictions = wknn_predict(points, point_targets, classes, k)
    return np.average(point_targets == predictions)


def compare_knns(
    points: np.ndarray,
    point_targets: np.ndarray,  # I changed this to point targets - Steinarr.
    classes: list
):
    # Remove if you don't go for independent section
    knn_accs = []
    wknn_accs = []
    for k in range(1, points.shape[0]):
        knn_accs.append(knn_accuracy(points, point_targets, classes, k))
        wknn_accs.append(wknn_accuracy(points, point_targets, classes, k))
    plt.plot(knn_accs, label="knn")
    plt.plot(wknn_accs, label="wknn")
    plt.legend()
    print("\n\n\t"
          "The weighted network performs better as k is large because of the weights.\n\t"
          "When k is very large, knn isn't doing much other than guessing the most common\n\t"
          "class. Wknn on the other hand considers more than just that as k gets large. \n\t"
          "Wknn isn't putting much weight behind a point that is far away. So the closer points\n\t"
          "overwhelm the voting system. Thus, adding more points far away from x doesn't affect \n\t"
          "the outcome of the prediction.")
    plt.show()


if __name__ == "__main__":
    d, t, classes = load_iris()
    x, points = d[0,:], d[1:, :]
    x_target, point_targets = t[0], t[1:]

    print(f"\n\n{'-' * 20}\n\t Part 1.1 \n")
    print(f"{euclidian_distance(x, points[0])=}")
    print(f"{euclidian_distance(x, points[50])=}")

    print(f"\n\n{'-' * 20}\n\t Part 1.2 \n")
    print(f"{euclidian_distances(x, points)=}")

    print(f"\n\n{'-' * 20}\n\t Part 1.3 \n")
    print(f"{k_nearest(x, points, 1)=}")
    print(f"{k_nearest(x, points, 3)=}")

    print(f"\n\n{'-' * 20}\n\t Part 1.4 \n")
    print(f"{vote(np.array([0,0,1,2]), np.array([0,1,2]))=}")
    print(f"{vote(np.array([1,1,1,1]), np.array([0,1]))=}")

    print(f"\n\n{'-' * 20}\n\t Part 1.5 \n")
    print(f"{knn(x, points, point_targets, classes, 1)=}")
    print(f"{knn(x, points, point_targets, classes, 5)=}")
    print(f"{knn(x, points, point_targets, classes, 150)=}")

    print(f"\n\n{'-' * 20}\n\t Part 2.1 \n")
    d, t, classes = load_iris()
    (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)
    print(f"{knn_predict(d_test, t_test, classes, 10)=}")
    print(f"{knn_predict(d_test, t_test, classes, 5)=}")

    print(f"\n\n{'-' * 20}\n\t Part 2.2 \n")
    print(f"{knn_accuracy(d_test, t_test, classes, 10)=}")
    print(f"{knn_accuracy(d_test, t_test, classes, 5)=}")

    print(f"\n\n{'-' * 20}\n\t Part 2.3 \n")
    print(f"{knn_confusion_matrix(d_test, t_test, classes, 10)=}")
    print(f"{knn_confusion_matrix(d_test, t_test, classes, 20)=}")

    print(f"\n\n{'-' * 20}\n\t Part 2.4 \n")
    print(f"{best_k(d_train, t_train, classes)=}")

    print(f"\n\n{'-' * 20}\n\t Part 2.5 \n")
    knn_plot_points(d, t, classes, 3)

    print(f"\n\n{'-' * 20}\n\t Part B.1 \n")
    targets = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    distances = np.array([2, 5, 3, 4, 2])
    print(f"{weighted_vote(targets, distances, classes)=}")

    print(f"\n\n{'-' * 20}\n\t Part B.2 \n")
    print(f"{wknn(x, points, point_targets, classes, 1)=}")
    print(f"{wknn(x, points, point_targets, classes, 5)=}")
    print(f"{wknn(x, points, point_targets, classes, 150)=}")

    print(f"\n\n{'-' * 20}\n\t Part B.3 \n")
    print(f"{knn_predict(d_test, t_test, classes, 10)=}")
    print(f"{knn_predict(d_test, t_test, classes, 5)=}")

    print(f"\n\n{'-' * 20}\n\t Part B.4 \n")
    compare_knns(d_test, t_test, classes)
