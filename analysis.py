import random
import math
import json
from muc import MUC
from typing import Set, Dict


class Shapley:
    """Compute Shapley values based on MUCs of
       a random forest."""

    def __init__(self, muc: MUC) -> None:
        self.muc = muc  # compute rf MUCs
        self.F = set(range(muc.rf.rfc.n_features_))  # feature set
        self.cache = dict()  # store computed MUCs
        self.X = muc.rf.rfc.X_test
        self.y = muc.rf.rfc.y_test

    def value(self, c: int, M: int) -> Dict[int, float]:
        """Compute M-Shapley values of class c.

        Args:
          c: Class index.
          M: Number of iterations of sampling.

        Returns:
          result: A dictionary with feature index as
                  key and Shapley value as item.
        """
        result = dict()
        len_f = len(self.F)
        for i in self.F:
            phi = 0
            for _ in range(M):
                s = set(filter(lambda x: random.randrange(2),
                               self.F - {i}))
                len_s = len(s)
                gain = self.omega(s | {i}, c) - self.omega(s, c)
                phi += gain / math.comb(len_f, len_s) / (len_f - len_s)
            result[i] = phi
        return result

    def omega(self,
              subset: Set[int],
              c: int) -> int:
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

    def compute_muc(self, file=''):
        """Compute MUCs and save to file."""
        try:
            with open(file, 'r') as f:
                self.cache = json.load(f)
            print(f'MUCs loaded from {file}.')
        except FileNotFoundError:
            print('Computing MUCs. This may take a while.')
            from tqdm import tqdm
            for i, x in enumerate(tqdm(self.X, desc='Computing MUCs')):
                muc = self.muc.muc(x, self.y[i])
                self.cache[i] = {
                    'muc': list(muc),
                    'y': self.y[i]
                }
            if file:
                with open(file, 'w') as f:
                    json.dump(self.cache, f)
                print(f'MUCs saved to {file}.')


class AdversarialSample:
    pass
