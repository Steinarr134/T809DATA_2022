# Author: Steinarr Hrafn
# Date: 4- sept 2022
# Project:
# Acknowledgements:


#
"""
Hann talaði um þessi föll til að hjalpa:

Numpy.random.multivariate_normal
Numpy.mean
Np.cov
Np.identity
Fyrir section 1.3 pdf() (probability density function)
Scipy.stats import multivariate_normal - creates distribution
.pdf(mean gildið)=hæsta gildið (erum i punktinum)"""

from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
        n: int,
        k: int,
        mean: np.ndarray,
        var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''

    return np.random.multivariate_normal(mean, var**2*np.identity(k), n)


def update_sequence_mean(
        mu: np.ndarray,
        x: np.ndarray,
        n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    return mu + (x - mu)/n

def gen_sequence_estimation(mean = np.array([0, 0, 0])):
    data = gen_data(100, 3, mean, 1)
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i], i+1))
    return estimates[1:]

def _plot_sequence_estimate():
    # generate sequence estimation in a seperate function so it can be reused later
    # mean = np.array([0, 1, -1])
    estimates = gen_sequence_estimation()
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    return np.square(y - y_hat)


def _plot_mean_square_error():
    mean = np.array([0, 1, -1])
    estimates = gen_sequence_estimation(mean)
    sq_e = _square_error(estimates, mean)
    sq_e = np.mean(sq_e, axis=1)
    plt.title("Mean Square Error")
    plt.plot(sq_e)
    plt.show()





# Naive solution to the independent question.

def gen_changing_data(
        n: int,
        k: int,
        start_mean: np.ndarray,
        end_mean: np.ndarray,
        var: float
) -> np.ndarray:

    # first create N datapoints with distributions around 0
    zero_centered = np.random.multivariate_normal(np.zeros((k)), var**2*np.identity(k), n)

    # then shift the datapoints according to the changing mean.
    shift = np.linspace(start_mean, end_mean, 500)
    return zero_centered + shift



def _plot_changing_sequence_estimate(m=75, show=True):

    # first
    def update_sequence_mean_modified(
            mu: np.ndarray,
            x: np.ndarray,
            n: int
    ) -> np.ndarray:
        '''Performs the mean sequence estimation update
        '''
        return mu + (x - mu)/min(n, m)

    def gen_sequence_estimation_modified(data):
        estimates = [np.array([0, 0, 0])]
        for i in range(data.shape[0]):
            estimates.append(update_sequence_mean_modified(estimates[i], data[i], i + 1))
        return estimates[:data.shape[0]]

    start_mean, end_mean = np.array([0, 1, -1]), np.array([1, -1, 0])
    data = gen_changing_data(500, 3, start_mean, end_mean, np.sqrt(3))
    estimates = gen_sequence_estimation_modified(data)
    plt.figure()
    plt.title(f"sequence estimation, changing mean, {m=}")
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')

    sq_e = _square_error(estimates, np.linspace(start_mean, end_mean, 500))
    sq_e = np.mean(sq_e, axis=1)

    plt.figure()
    plt.title(f"Mean Square Error, changing mean, {m=}")
    plt.plot(sq_e)
    # plt.ylim(0, 0.25)
    if show:
        plt.show()



if __name__ == "__main__":

    np.random.seed(1234)
    print(f"\n\n{'-' * 20}\n\t Part 1.1 \n")
    print(f"{gen_data(2, 3, np.array([0, 1, -1]), 1.3)=}")
    np.random.seed(1234)
    print(f"{gen_data(5, 1, np.array([0.5]), 0.5)=}")

    print(f"\n\n{'-' * 20}\n\t Part 1.2 \n")

    np.random.seed(1234)
    X = gen_data(300, 3, np.array([0, 1, -1]), np.sqrt(3))
    print(f"{np.mean(X, axis=0)=}")
    # scatter_3d_data(X)
    # bar_per_axis(X)

    print(f"\n\n{'-' * 20}\n\t Part 1.4 \n")
    
    mean = np.mean(X, 0)
    new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
    print(f"{update_sequence_mean(mean, new_x, X.shape[0])}")

    print(f"\n\n{'-' * 20}\n\t Part 1.5 \n")
    _plot_sequence_estimate()

    print(f"\n\n{'-' * 20}\n\t Part 1.6 \n")
    _plot_mean_square_error()

    print(f"\n\n{'-' * 20}\n\t Independent section \n")
    # _plot_changing_sequence_estimate(50, 0)
    # _plot_changing_sequence_estimate(150, 0)
    # plt.show()
    _plot_changing_sequence_estimate()

