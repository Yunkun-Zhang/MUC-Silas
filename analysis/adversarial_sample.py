import time
import numpy as np
from model.muc import MUC
from typing import Sequence

np.set_printoptions(formatter={'float': '{:.3f}'.format})


class AdversarialSample:
    """Generate adversarial sample based on
       MUCs of a random forest."""

    def __init__(self, muc: MUC, verbose: bool = False):
        """Create adversarial sample generator.

        Args:
          muc: MUC object.
          verbose: Whether to print log information.
        """
        self.muc = muc
        self.verbose = verbose
        self.__n_features = muc.rf.rfc.n_features_
        self.F = set(range(self.__n_features))
        self.binary = self.muc.rf.rfc.binary
        self.binary_list = sum((lst for lst in self.binary.values()), [])
        self.binary_mask = np.array([False] * self.__n_features)
        self.binary_mask[self.binary_list] = True
        # search step prop to mean of each feature
        self.kappa = self.__get_step()

    @staticmethod
    def distance(x1, x2):
        """Compute distance between x_org and x_adv.

        .. math::
            dist = 1/m \\times \\sum_{j=1}^m \\| x_j^{org} - x_j^* \\|

        """
        return np.linalg.norm(x1 - x2, 1) / len(x1)

    def fine_grained_binary_search(
            self,
            x_org: np.ndarray,
            y: int,
            theta: np.ndarray,
            unvisited: Sequence[int],
            alpha=0.2, epsilon=1e-5) -> float:
        """Fine-grained and binary search. (Alg 2)

        Args:
          x_org: Data instance.
          y: Ground truth for x_org.
          theta: Search direction.
          unvisited: Unvisited feature indices.
          alpha: Increasing/decreasing ratio.
          epsilon: Stopping tolerance.

        Returns:
          v_out: Vector multiplied by theta is the final
                 closer adversarial sample.
        """
        unvisited = list(set(unvisited))
        norm = np.linalg.norm(theta)
        # unvisited features are not extended
        theta[unvisited] = 0
        theta = theta / norm
        v_in = v_out = norm
        # lines 3-5
        # TODO: how to deal with input with same class?
        while self.muc.predict(self.__org2adv(x_org, theta, v_out, norm)) == y:
            v_in = v_out
            v_out = v_in / (1 - alpha)
        while self.muc.predict(self.__org2adv(x_org, theta, v_in, norm)) != y:
            v_out = v_in
            v_in = v_out * (1 - alpha)
        # lines 6-13
        while v_out - v_in > epsilon:
            v_mid = (v_out + v_in) / 2
            if self.muc.predict(self.__org2adv(x_org, theta, v_mid, norm)) != y:
                v_out = v_mid
            else:
                v_in = v_mid
        return v_out

    def opt_adv_sample(
            self,
            X: Sequence[float],
            y: int,
            num_itr: int = 1,
            num_samples: int = 1,
            beta: float = 0.001,
            eta: float = 0.01) -> np.ndarray:
        """Generate the optimized adversarial sample. (Alg 3)

        Args:
          X: Data instance, 1D array. Only accept numeric data now.
          y: Ground truth of X.
          num_itr: Number of iterations.
          num_samples: Number of samples picked each iteration.
          beta: Smoothing parameter.
          eta: Step size.

        Returns:
          An adversarial sample, 1D array.
        """
        assert len(X) == self.__n_features and isinstance(X[0], (int, float)), \
            f'data instance must be 1D array with size {self.__n_features}'

        start_time = time.time()
        tau, unvisited = self.__get_region(X, y)
        t1 = time.time() - start_time
        if self.verbose:
            print(f'Good tau found:     {tau}. ({t1:.3f}s)')

        start_time = time.time()
        X = np.array(X)
        theta_0 = self.__get_theta(X, y, tau, unvisited, num_samples)
        t2 = time.time() - start_time
        if self.verbose:
            print(f'Best theta found:   {theta_0}. ({t2:.3f}s)')

        start_time = time.time()
        opt_sample = self.__optimize(X, y, theta_0, unvisited, num_itr, beta, eta)
        t3 = time.time() - start_time
        if self.verbose:
            print(f'Optimized sample found. ({t3:.3f}s)')

        return opt_sample

    def __get_region(self, X, y):
        """Alg 3, lines 1-5, get adversarial region."""
        # TODO: how to deal with nominal features?
        tau = [0 for _ in self.F]
        muc = self.muc.muc(X, y, tau)
        unvisited = self.F
        while muc:
            for f in muc:
                tau[f] += self.kappa[f]
                unvisited -= {f}
            muc = self.muc.muc(X, y, tau)
        tau = np.array(tau)
        return tau, unvisited

    def __get_theta(self, X, y, tau, unvisited, num_samples):
        """Alg 3, lines 6-13, generate samples and get the best one."""
        lamda_min = 2147483647
        theta_0 = np.zeros(X.size)
        while np.all(theta_0 == 0):
            samples = np.random.uniform(
                X - tau, X + tau, size=(num_samples, self.__n_features)
            )
            samples = self.__force_one_hot(samples)
            # loop for theta_0
            for x in samples:
                if np.all(x == X):
                    continue
                if self.muc.predict(x) != y:
                    x = np.array(x)
                    theta = x - X  # adv sample direction
                    lamda = self.fine_grained_binary_search(
                        X, y, theta, unvisited
                    )
                    if lamda_min > lamda:
                        lamda_min = lamda
                        theta_0 = theta
        return theta_0

    def __optimize(self, X, y, theta_0, unvisited, num_itr, beta, eta):
        """Alg 3, lines 14-19, iter to compute opt sample."""
        # TODO: how is mu generated?
        mu = self.kappa / np.max(self.kappa) * 0.01
        mu[self.binary_list] = 0
        theta_t = theta_0
        if self.verbose:
            print(f'Before opt:         {X + theta_0}')
            print(f'Distance:           {self.distance(X, X + theta_0)}')
        for _ in range(num_itr):
            u = np.random.normal(size=X.size)
            u[self.binary_list] = 0
            u = u / np.linalg.norm(u)
            theta_ts = theta_t + u * mu * beta
            gs = self.fine_grained_binary_search(X, y, theta_ts, unvisited)
            g = self.fine_grained_binary_search(X, y, theta_t, unvisited)
            g_hat = (gs - g) * u / beta
            theta_t = theta_t - g_hat * eta
        gt = self.fine_grained_binary_search(X, y, theta_t, unvisited)
        norm = np.linalg.norm(theta_t)
        theta_t = theta_t / norm
        opt_sample = self.__org2adv(X, theta_t, gt, norm)
        return opt_sample

    def __get_step(self, ratio=0.1):
        """Compute self.kappa from the mean of test data."""
        X = self.muc.rf.rfc.X_test
        kappa = np.mean(X, axis=0) * ratio
        kappa[self.binary_list] = 1
        return kappa

    def __org2adv(self, x_org, theta, dist, norm):
        x_adv = np.where(self.binary_mask,
                         x_org + theta * norm,
                         x_org + theta * dist)
        return x_adv

    def __force_one_hot(self, samples):
        if len(samples.shape) == 1:
            samples = samples[np.newaxis, :]
        samples[:, self.binary_list] = np.where(samples[:, self.binary_list] <= 0.5, 0, 1)
        for group, indices in self.binary.items():
            am = np.stack([np.arange(samples.shape[0]), np.argmax(samples[:, indices], axis=-1)])
            samples[:, indices] = 0
            samples[am[0], am[1]] = 1
        if len(samples) == 1:
            samples = np.reshape(samples, samples.shape[1])
        return samples
