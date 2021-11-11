import time
import os
import random
import math
import json
from model.muc import MUC
from utils import _parallel, _compute_muc
from typing import Dict


class Shapley:
    """Compute Shapley values based on MUCs of
       a random forest."""

    def __init__(self, muc: MUC, verbose: bool = False) -> None:
        """Create a Shapley value computer.

        Args:
          muc: MUC object.
          verbose: Whether to print log information.
        """
        self.muc = muc  # compute rf MUCs
        self.verbose = verbose
        self.F = set(range(muc.rf.rfc.n_features_))  # feature set
        self.cache = dict()  # store computed MUCs

    @staticmethod
    def __power_set(s):
        x = len(s)
        s = list(s)
        for i in range(1 << x):
            yield set([s[j] for j in range(x) if (i & (1 << j))])

    def __sample(self, i, M=None):
        if M is None:
            for s in self.__power_set(self.F - {i}):
                yield s
        else:
            for _ in range(M):
                yield set(filter(
                    lambda x: random.randrange(2),
                    self.F - {i})
                )

    def value(self, c: int, M: int = None) -> Dict[int, float]:
        """Compute M-Shapley values of class c. (Alg 1)

        Args:
          c: Class index.
          M: Number of iterations of sampling.

        Returns:
          result: A dictionary with feature index as
                  key and Shapley value as item.
        """
        if self.verbose:
            print(f'Computing M-Shapley values for class {c}.', end=' ')
        start = time.time()
        result = dict()
        len_f = len(self.F)
        for i in self.F:
            phi = 0
            for s in self.__sample(i, M):
                len_s = len(s)
                gain = self.__omega(s | {i}, c) - self.__omega(s, c)
                phi += gain / math.comb(len_f, len_s) / (len_f - len_s)
            result[i] = phi
        if self.verbose:
            print(f'({time.time() - start:.3f}s)')
        return result

    def __omega(self, subset, c):
        """Calculate worth of a feature subset.

        Args:
          subset: Subset of feature indices.
          c: Class index.

        Returns:
          Omega: Worth value.
        """
        Omega = 0
        for x in self.cache:
            muc = set(self.cache[x]['muc'])
            label = self.cache[x]['y']
            if subset.issubset(muc):
                Omega = Omega + 1 if label == c else Omega - 1
        return Omega

    def compute_muc(self, X, y, save_file=''):
        """Compute MUCs and save to file."""
        if os.path.exists(save_file):
            with open(save_file, 'r') as f:
                self.cache = json.load(f)
            if self.verbose:
                print(f'MUCs loaded from {save_file}.')
        else:
            if self.verbose:
                print('Computing MUCs. This may take a while.')
            s = time.time()
            res = _parallel(_compute_muc, list(zip(X, y)),
                            processes=8, muc=self.muc)
            for i, muc in enumerate(res):
                self.cache[i] = {
                    'muc': list(muc),
                    'y': y[i]
                }
            if save_file:
                with open(save_file, 'w') as f:
                    json.dump(self.cache, f)
                if self.verbose:
                    print(f'MUCs saved to {save_file}. ({time.time() - s:.3f}s)')
