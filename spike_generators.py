import numpy as np

class GutniskyUnivariateSpikeGenerator:
    '''
    https://journals.physiology.org/doi/full/10.1152/jn.00518.2009
    '''

    def __init__(self) -> None:
        pass

    def get_spike_train(self, r, spike_train_autocov):
        '''
        Parameters:
        ----------
        r: float
            Neuron's mean spiking rate.

        spike_train_autocov: ndarray(shape = len_spike_train)
            Autocovariance of the spike train (c(tau) from the paper).

        Returns:
        ----------
        x: ndarray
            Spike train of the same length as spike_train_autocov.

        times: ndarray
            Interspike time intervals.
        '''
        # Get theta, the threshold for AR process, 
        # by finding: r = integrate(y_min, y_max) - integrate(y_min, theta) | Equation 12
        theta = self.get_theta(r)

        # Get rho_tau, autocovariance of bivariate Gaussians, from spike autocovariance
        rho_tau = spike_train_autocov + r ** 2

        # Do a linear search for R_tau
        R_tau = []

        possible_rtau = np.arange(-9, 10) * 0.1
        for i in range(len(spike_train_autocov)):
            candidates = []
            for rtau_val in possible_rtau:
                candidates.append(self.compute_rho_tau(rtau_val, theta))
            R_tau.append(possible_rtau[np.argmin(np.abs(rho_tau[i] - np.array(candidates)))])

        R_tau = np.array(R_tau)

        # Compute AR params
        A, sigma = self.generate_AR_params(R_tau)

        # Generate AR process with given params
        # The paper uses AR(k), but AR(1) looks more stable

        # Version from the paper: AR(k)
        #y = [1] * A.shape[0]
        #for i in range(len(spike_train_autocov)):
        #    y.append(np.sum(A * np.array(y[-len(spike_train_autocov) + 1:]).reshape((-1, 1))) + np.random.normal(0, sigma))

        # Stable version: AR(1)
        y = [1]
        for i in range(1000):
            yt = y[-1] * A[0] + np.random.normal(0, sigma)
            y.append(yt[0])

        # Threshold
        x = (np.array(y) > theta).astype(int)

        # Get interspike times
        times = self.get_intespike_times(x)

        return x, times, y
    
    def get_theta(self, r):
        # Generate y(t) ~ N(0, 1)
        y = np.random.normal(size = 2000)
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
    
    def compute_rho_tau(self, r_tau, theta):
        # Compute integral from Eqaution 13
        y = np.random.normal(size = 2000)
        y = np.sort(y)
        y = y[y > theta]

        rho = 0
        const = 1 / (2 * np.pi * np.sqrt(1 - r_tau ** 2))
        const_exp = 2 * (1 - r_tau ** 2)

        for i in range(len(y) - 1):
            rho += const * np.exp(-(y[i] ** 2 + y[i+1] ** 2 - 2 * r_tau * y[i] * y[i+1]) / const_exp) * (y[i+1] - y[i])

        return rho
    
    def generate_AR_params(self, autocovariance_vector):
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
        times = []

        t = 0
        for i in range(len(train)):
            if train[i] == 1:
                times.append(i - t)
                t = i

        return np.array(times)[1:]
    


class KaulakysUnivariateSpikeGenerator:
    '''
    https://www.researchgate.net/profile/Vygintas-Gontis/publication/235753842_Long-range_stochastic_point_processes_with_the_power_law_statistics/links/02e7e52622e0350178000000/Long-range-stochastic-point-processes-with-the-power-law-statistics.pdf?origin=publication_detail
    '''

    def __init__(self) -> None:
        pass

    def get_spike_train(self, r, spike_train_length, sigma = 0.01):
        '''
        Parameters:
        ----------
        r: float
            Neuron's mean spiking rate.

        spike_train_length: int
            Length of the train.

        sigma: float
            Standard deviation of white noise.

        Returns:
        ----------
        spike_train: ndarray
            Spike train.

        interevent_time: ndarray
            Time intervals between spikes
        '''
        interevent_time = np.zeros(spike_train_length)
        interevent_time[0] = r

        # Generate interevent times based on Equation 13
        for i in range(1, spike_train_length):
            interevent_time[i] = interevent_time[i-1] - sigma ** 2 / 2 + sigma * np.random.normal(0, 1)

        # Get interspike time
        spike_train = []
        for t in interevent_time:
            interval = [0] * int(t) + [1]
            spike_train.extend(interval)

        spike_train = np.array(spike_train)[:spike_train_length]

        return spike_train, interevent_time[:sum(spike_train) - 1]