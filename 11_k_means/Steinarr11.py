# Author:
# Date:
# Project:
# Acknowledgements:
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results


def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''
    ret = np.empty((X.shape[0], Mu.shape[0]))
    for i, s in enumerate(X):
        for j, p in enumerate(Mu):
            ret[i, j] = np.sqrt(np.sum(np.square(np.subtract(s, p))))
    return ret

""" Section 1.1"""
if __name__ == "__main__":
    a = np.array([
        [1, 0, 0],
        [4, 4, 4],
        [2, 2, 2]])
    b = np.array([
        [0, 0, 0],
        [4, 4, 4]])
    print(f"{distance_matrix(a, b)=}")

def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    ret = np.zeros(dist.shape)
    ret[np.arange(dist.shape[0]), np.argmin(dist, axis=1)] = 1
    return ret

"""Section 1.2"""
if __name__ == "__main__":
    dist = np.array([
        [1,   2,   3],
        [0.3, 0.1, 0.2],
        [7,  18,   2],
        [2, 0.5,   7]])
    print(f"{determine_r(dist)=}")

def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    '''
    return np.sum(R*dist)/R.shape[0]


""" Section 1.3 """
if __name__ == "__main__":
    dist = np.array([
        [1,   2,   3],
        [0.3, 0.1, 0.2],
        [7,  18,   2],
        [2, 0.5,   7]])
    R = determine_r(dist)
    print(f"{determine_j(R, dist)=}")


def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''
    # print(f"{X.shape=}, {R.shape=}")
    # return np.divide(np.matmul(X.T, R), np.sum(R, axis=0)).T
    ret = np.zeros((R.shape[1], X.shape[1]))
    for k in range(R.shape[1]):
        # print(f"{R[:, k].shape=}, * {X.shape=}")
        # print(f"{np.sum(R[:, k]*X, axis=1).shape=}, {np.sum(R[:, k])=}")
        ret[k, :] = np.sum(R[:, k]*X.T, axis=1)/np.sum(R[:, k])
    return ret

""" Section 1.4 """
if __name__ == "__main__":
    X = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
        [1, 1, 0]
        ])
    Mu = np.array([
        [0.0, 0.5, 0.1],
        [0.8, 0.2, 0.3]])
    R = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [1, 0]
        ])
    print(f"{update_Mu(Mu, X, R)=}")

def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    Js = []

    for i in range(num_its):
        dist = distance_matrix(X_standard, Mu)
        R = determine_r(dist)
        Js.append(determine_j(R, dist))
        # print(f"{dist.shape=}, {Mu.shape=}, {X_standard.shape=}")
        Mu = update_Mu(Mu, X_standard, R)
    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    return Mu, R, Js



""" Section 1.5"""
if __name__ == "__main__":
    X, y, c = load_iris()
    print(f"{k_means(X, 4, 10)=}")


def _plot_j(savename=None):
    X, y, c = load_iris()
    pMu, R, Js = k_means(X, 4, 10)
    plt.plot(Js)
    plt.xlabel("Iterations")
    plt.ylabel("J")
    if savename is not None:
        plt.savefig(savename)
    plt.show()
    

def _plot_multi_j(savename=None):
    X, y, c = load_iris()
    plt.figure(figsize=[6, 8])
    for i, k in enumerate([2, 3, 5, 10]):
        pMu, R, Js = k_means(X, k, 10)
        plt.subplot(4, 1, i+1)
        plt.plot(Js)
        plt.title(str(k) + " classes", y=1.0, pad=-14)
        plt.xlabel("Iterations")
        plt.ylabel("J")
    if savename is not None:
        plt.savefig(savename)
    plt.show()

# """Section 1.6 & 1.7"""
# if __name__ == "__main__":
#     _plot_j("11_k_means/1_6_1.png")
#     _plot_multi_j("11_k_means/1_7_1.png")

def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''
    Mu, R, Js = k_means(X, len(classes), num_its)
    mat = np.zeros((len(classes), len(classes)))
    for i in range(t.shape[0]):
        mat[np.where(np.array(classes) == t[i]), np.argmax(R[i, :])]+=1
    print(mat)
    class_order = np.argmax(mat, axis=0)
    predictions = []
    for i in range(t.shape[0]):
        predictions.append(class_order[np.argmax(R[i, :])])
    return predictions



def _iris_kmeans_accuracy():
    X, y, c = load_iris()
    predictions = k_means_predict(X, y, c, 5)
    correct = 0
    for i in range(y.shape[0]):
        if predictions[i] == y[i]:
            correct += 1
    print("kmeans accuracy: ", correct/y.shape[0])

"""Section 1.9 & 1.10"""
if __name__ == "__main__":
    X, y, c = load_iris()
    print(c)
    print(f"{k_means_predict(X, y, c, 5)=}")
    _iris_kmeans_accuracy()

def _my_kmeans_on_image():
    image, (w, h) = image_to_numpy()
    X = image.reshape(w*h, 3)
    k_means(X, 7, 5)




def plot_image_clusters(n_clusters: int, save=0):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    image, (w, h) = image_to_numpy()
    X = image.reshape(w*h, 3)

    kmeans = KMeans(n_clusters)
    kmeans.fit(X)
    plt.figure(figsize=[7, 3])
    plt.subplot(121)
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot(122)
    # uncomment the following line to run
    plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    if save:
        plt.savefig(f"11_k_means/2_1_{save}.png")
    plt.show()

""" Section 2.1"""
# if __name__ == "__main__":
#     # _my_kmeans_on_image()
#     for i, c in enumerate([2, 5, 10, 20]):
#         plot_image_clusters(c, i+1)



def _gmm_info():
    ...


def _plot_gmm():
    ...

"""Independent"""

if __name__ == "__main__":

    def get_weather_data():
        temps = []
        high_temp = []
        low_temp = []
        top_temp = []
        bottom_temp = []
        pressures = []
        clouds = []
        rains = []
        suns = []
        winds = []
        month = []
        with open("08_SVM/medalvedur_rvk.txt", 'r') as f:
            for line in f.readlines():
                line = line.strip()
                stuff = line.split("\t")
                stuff = [float(s) for s in stuff]
                temps.append(stuff[3])
                high_temp.append(stuff[4])
                top_temp.append(stuff[5])
                low_temp.append(stuff[7])
                bottom_temp.append(stuff[8])
                pressures.append(stuff[14])
                rains.append(stuff[11])
                clouds.append(stuff[15])
                suns.append(stuff[16])
                winds.append(stuff[17])
                month.append(stuff[2])
        return np.vstack([temps, high_temp, top_temp, low_temp, bottom_temp, suns,  winds,  rains, clouds]).T, np.array(month).T

    X, y = get_weather_data()
    print(X, y)
    kmeans = KMeans(4).fit(X)
    
    res = np.zeros((12, kmeans.n_clusters))
    for i in range(y.shape[0]):
        res[int(y[i])-1, int(kmeans.labels_[i])] += 1
    # res = confusion_matrix(y, kmeans.labels_)
    print(res.T)
    print(kmeans.cluster_centers_)