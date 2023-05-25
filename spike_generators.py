import numpy as np
from scipy.optimize import minimize

class GutniskyUnivariateSpikeGenerator:
    '''
    Generates a spike train with given autocovariance.
    https://journals.physiology.org/doi/full/10.1152/jn.00518.2009
    '''

    def get_spike_train(self, r, spike_train_autocov, len_spike_train):
        '''
        Generate a spike train of given length and autocovariance.

        Parameters:
        ----------
        r: float
            A neuron's mean spiking rate.

        spike_train_autocov: np.ndarray
            Autocovariance of the spike train (c(tau) from the paper).

        len_spike_train: int
            Length of the spike train to generate.

        Returns:
        ----------
        train: np.ndarray
            Spike train of the same length as spike_train_autocov (x from the paper).

        y: np.ndarray
            AR(1) process which is thresholded to obtain train.
        '''
        # Get theta, the threshold for AR process, 
        # by finding: r = integrate(y_min, y_max) - integrate(y_min, theta) | Equation 12
        theta = self.get_theta(r)

        # Get rho_tau, autocovariance of bivariate Gaussians, from spike autocovariance
        rho_tau = spike_train_autocov + r ** 2

        # Do a linear search for R_tau, autocovariance of a univariate Gaussian
        # (Tried scipy.optimize.minimize, unstable)
        R_tau = [1]

        possible_rtau = np.arange(-9, 10) * 0.1
        for i in range(1, len(rho_tau)):
            candidates = []
            for rtau_val in possible_rtau:
                candidates.append(self.compute_rho_tau(rtau_val, theta, i))
            R_tau.append(possible_rtau[np.argmin(np.abs(rho_tau[i] - np.array(candidates)))])

        R_tau = np.array(R_tau)

        # Compute AR params
        A, sigma = self.generate_AR_params(R_tau)

        # Generate an AR process with given params
        y = [1] * A.shape[0]
        for i in range(len_spike_train + A.shape[0]):
            y.append(np.sum(A * np.array(y[-A.shape[0]:]).reshape((-1, 1))) + np.random.normal(0, sigma))
        y = y[A.shape[0]:]

        # Threshold
        train = (np.array(y) > theta).astype(int)

        return train, y
    
    def get_theta(self, r):
        '''
        Compute theta, the threshold for the AR process (Eq 12).

        Parameters:
        ----------
        r: float
            A neuron's mean spiking rate.

        Returns:
        ----------
        theta: float
            Threshold.
        '''
        # Generate y(t) ~ N(0, 1)
        y = np.random.normal(size = 5000)
        y = np.sort(y)
        dy = np.concatenate([[0], np.diff(y)])

        # Compute r by Equation 12, but from -inf to inf
        r_int = 1 / np.sqrt(2 * np.pi) * np.sum(np.exp(-y ** 2 / 2) * dy)

        # Provided r is the integral from theta to inf
        # Subtract r from r_int to get the integral from -inf to theta
        si = r_int - r

        # Compute integrals (-inf; -inf + 1), (-inf; -inf + 2), ..., (-inf; -inf + theta)
        r_int_cum = np.cumsum(1 / np.sqrt(2 * np.pi) * np.exp(-y ** 2 / 2) * dy)

        # Find theta
        theta = float(y[np.argwhere(si < r_int_cum)[0]])

        return theta
    
    def compute_rho_tau(self, R_tau, theta, tau):
        '''
        Computes rho(tau), the integral from Equation 13.

        Parameters:
        ----------
        R_tau: np.ndarray
            Autocovariance of the AR process.

        theta: float
            Lower bound of the integral.

        tau: float
            Value of tau with which R_tau was computed.

        Returns:
        ----------
        rho_tau: float
            The integral.
        '''
        # Compute the integral from Eqaution 13
        y = np.random.normal(size = 5000)
        y = np.sort(y)
        y = y[y > theta]

        rho_tau = 0

        for i in range(1, len(y) - tau):
            const = 1 / (2 * np.pi * np.sqrt(1 - R_tau ** 2))
            const_exp = 2 * (1 - R_tau ** 2)
            dadb = (y[i] - y[i-1]) * (y[i + tau] - y[i + tau - 1])
            rho_tau +=  dadb * const * np.exp(-(y[i] ** 2 + y[i + tau] ** 2 - 2 * R_tau * y[i] * y[i + tau]) / const_exp)

        return rho_tau
    
    def generate_AR_params(self, autocovariance_vector):
        '''
        Computes the vector of coefficients and variance of noise of an AR process with the given autocovariance
        structure.

        Parameters:
        ----------
        autocovariance_vector: np.ndarray
            Autocovariance [1, <yt, yt-1>, <yt, yt-2>, ...]

        Returns:
        ----------
        A: np.ndarray
            Vector of coefficients.

        sigma: float
            White noise variance.
        '''
        autocovariance_vector = autocovariance_vector.reshape((-1, 1))
        L = autocovariance_vector.shape[0] - 1
        
        # Construct the T matrix
        T = np.zeros((L, L))

        # Iterate through i = 0...L-1 and create the i-th upper diagonal of T
        # counting from the main diagonal
        for i in range(L):
            T = T + np.diag(np.ones((L - i)) * autocovariance_vector[i], i)

        # Apply symmetric transform
        T = T + T.T - np.diag(np.diag(T))

        # Compute A, the coefficient matrix
        A = np.linalg.pinv(T) @ autocovariance_vector[1:, :]

        # Compute variance of gaussian noise
        sigma = float(autocovariance_vector[0, :] - A.T @ autocovariance_vector[1:, :])
        if sigma < 0:
            sigma = 1
        
        return A, sigma
    
    def get_intespike_times(self, train):
        '''
        A helper function to compute interspike times for a given train.

        Parameters:
        ----------
        train: np.ndarray
            Spike train.

        Returns:
        times: np.ndarray
            Interspike intervals.
        '''
        times = []

        t = 0
        for i in range(len(train)):
            if train[i] == 1:
                times.append(i - t)
                t = i

        return np.array(times)[1:]
    


class KaulakysUnivariateSpikeGenerator:
    '''
    Generates interspike intervals with 1/f psd.
    https://www.researchgate.net/profile/Vygintas-Gontis/publication/235753842_Long-range_stochastic_point_processes_with_the_power_law_statistics/links/02e7e52622e0350178000000/Long-range-stochastic-point-processes-with-the-power-law-statistics.pdf?origin=publication_detail
    '''

    def __init__(self) -> None:
        pass

    def get_spike_train(self, r, number_of_intervals, sigma = 0.01):
        '''
        Parameters:
        ----------
        r: float
            A neuron's mean spiking rate.

        number_of_intervals: int
            Number of intervals to generate.

        sigma: float
            Standard deviation of white noise. Bigger values correspond to y_{t} and y_{t+1} being less similar.

        Returns:
        ----------
        intervals: np.ndarray
            Time intervals between spikes
        '''
        intervals = np.zeros(number_of_intervals)
        intervals[0] = r

        # Generate interevent times based on Equation 13
        for i in range(1, number_of_intervals):
            intervals[i] = intervals[i-1] - sigma ** 2 / 2 + sigma * np.random.normal(0, 1)

        return intervals