import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class McEstimator:
    """
        The class is designed to calculate moments errors of distributions
        with given uncertainties by the Monte-Carlo method.
    """
    def __init__(self, filepath=None, distribution=None,
                 errors=None, fromfile=True, columns=("restored", "restored_errors")):
        """
        :param filepath: string - csv data file filepath. The method reads data from csv file if fromfile=True.
                            File must contain the same columns as in 'columns' input.
        :param distribution: np.ndarray or any convertable - The given distribution if fromfile=False.
        :param errors: np.ndarray or any convertable - Uncertainties of given distribution if fromfile=False.
        :param fromfile: boolean, default=True - the method reads data from file if True
                                                    and from 'distributon' input else.
        :param columns: couple, optimal - a pair of columns that contains distribution and its errors
                                            in .csv file. ("restored", "restored_errors") as default.
        """
        if fromfile:
            _data = pd.read_csv(filepath)
            self._distribution = _data[columns[0]]
            self._errors = _data[columns[1]]
        else:
            if not isinstance(distribution, np.ndarray):
                distribution = np.array(distribution)
            if not isinstance(errors, np.ndarray):
                errors = np.array(errors)
            self._distribution = distribution
            self._errors = errors

    @staticmethod
    def mean(distribution, x=None):
        """
        The method calculates an expected value E(X) of random variable X
                                        with the given distribution P(X).
            E(X) = sum(E(Xi) * P(Xi))
        :param distribution:  np.ndarray - Given distribution
        :param x: np.ndarray, optional - Bins of distribution. Range(len(distribution)) as default.
        :return: float - An expected value of X
        """
        if not x:
            x = np.array(list(range(len(distribution))))
        return distribution @ x

    @staticmethod
    def var(distribution, x=None):
        """
        The method calculates a variance V(X) of random variable X
                                        with given distribution P(X).
                                        V(X) = sum(P(Xi) * (Xi - E(X))^2)
        :param distribution: np.ndarray - Given distribution
        :param x: np.ndarray, optional - Bins of distribution. Range(len(distribution)) as default.
        :return: float - A variance of X
        """
        if not x:
            x = np.array(list(range(len(distribution))))
        return np.power((x - distribution @ x), 2) @ distribution

    def estimate_mean_and_variance(self, size_=100000, print_result=True):
        """
        The method estimates confidence intervals of expected value and variance
                                        of distribution using Monte-carlo method.
        :param size_: int, optional - The sample size for Monte-Carlo simulation
        :param print_result: boolean, optional - The method prints result if True
                                                          and just return it else
        :return: couple of couple of floats
                (mean, mean_error), (variance, variance_error) - estimation results
        """

        n = self._distribution.size
        _samples = np.empty(shape=(size_, n))
        _means = []
        _vars = []
        for i in range(n):
            _sample = np.random.normal(loc=self._distribution[i],
                                       scale=self._errors[i],
                                       size=size_)
            _samples[:, i] = _sample

        for d in _samples:
            _means.append(self.mean(d))
            _vars.append(self.var(d))

        _mean, _mean_error = np.mean(_means), np.std(_means)
        _var, _var_error = np.mean(_vars), np.std(_vars)

        if print_result:
            print("\n####################################\n")
            print("Monte-carlo estimation results:")
            print(f"mean value: {_mean:.3f} +- {_mean_error:.3f}")
            print(f"variance: {_var:.3f} +- {_var_error:.3f}")
            print("\n###################################\n")
        return (_mean, _mean_error), (_var, _var_error)


def mean(distribution, x=None):
    if not x:
        x = np.array(list(range(len(distribution))))
    return distribution @ x


def var(distribution, x=None):
    if not x:
        x = np.array(list(range(len(distribution))))
    return np.power((x - distribution @ x), 2) @ distribution


if __name__ == "__main__":
    data = pd.read_csv("data_csv/Rf256_2014_restored_12_2021.csv")
    restored = data["restored"].to_numpy()
    errors = data["restored_errors"].to_numpy()
    size = 100000
    print("[", end="")
    [print(e, ",", end=" ", sep="") for e in restored]
    print("]")
    print("[", end="")
    [print(e, ",", end=" ", sep="") for e in errors]
    print("]")
    samples = np.empty(shape=[size, restored.size])

    for i in range(restored.size):
        sample = np.random.normal(loc=restored[i], scale=errors[i], size=size)
        sample[sample < 0] = 0
        samples[:, i] = sample
    #     plt.scatter([i]*size, sample, alpha=0.1)
    #
    # plt.errorbar(list(range(10)), restored, yerr=errors)
    # plt.show()

    means = []
    vars = []

    for d in samples:
        means.append(mean(d))
        vars.append(var(d))

    print(mean(restored), var(restored))

    print(f"{np.mean(means):.3f}+-{np.std(means):.3f}")
    print(f"{np.mean(vars):.3f}+-{np.std(vars):.3f}")

    mc = McEstimator("data_csv/Rf256_2014_restored_12_2021.csv")
    mc.estimate_mean_and_variance()

    # mean = 4.32
    # var = 3.28

    print(var(restored))
