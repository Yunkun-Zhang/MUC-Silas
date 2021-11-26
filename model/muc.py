from z3 import *
from model.silas import DT, RFC
from typing import List, Set, Sequence, Dict

set_option(rational_to_decimal=True)


def _abs(x):
    """Return the absolute value of x."""
    return If(x >= 0, x, -x)


def _argmax(lst: List) -> int:
    """Get argmax from a list."""
    m = lst[0]
    j = 0
    for i, x in enumerate(lst):
        flag = x > m
        j = If(flag, i, j)
        m = If(flag, x, m)
    return j


def _eq_bound(lst, val, tau=None, nominal_features=None):
    """Return a Z3 expression.

    For numerical feature: val - tau <= x <= rng + val.
    For nominal feature: x in nominal_features.
    """
    if tau is None:
        return [v1 == v2 for v1, v2 in zip(lst, val)]
    return [_in(lst[i], nominal_features[i]) if i in nominal_features else
            _abs(lst[i] - val[i]) <= tau[i]
            for i in range(len(val))]


def _sum_weight(lists: List[List[ArithRef]], weights: List[int] = None):
    """Add up some Real lists with weight."""
    res = [0] * len(lists[0])
    if weights is None:
        weights = [1] * len(lists)
    for ind, lst in enumerate(lists):
        for i, val in enumerate(lst):
            res[i] = res[i] + val * weights[ind]
    return res


def _in(x, lst):
    """Whether x appears in lst."""
    return Or([x == val for val in lst])


def _implies(p, x):
    """Connect each bool variable to an expression."""
    return [Implies(prop, expr) for prop, expr in zip(p, x)]


class Tree:
    def __init__(self, dt: DT) -> None:
        """Build CNF from a decision tree."""
        self.dt = dt

    def __call__(self, x: List[ArithRef]) -> List:
        """Return the output of a decision tree."""
        t = self.dt

        def dfs(node):
            if t.children_left[node] == -1:
                return [int(val) for val in t.value[node]]
            feature = t.feature[node]
            threshold = t.threshold[node]
            left = dfs(t.children_left[node])
            right = dfs(t.children_right[node])
            if isinstance(threshold, list):
                flag = _in(x[feature], threshold)
            else:
                flag = x[feature] <= threshold
            return [If(flag, l, r) for l, r in zip(left, right)]

        return dfs(0)


class RandomForest:
    def __init__(self, rfc: RFC) -> None:
        """Build CNF from a random forest."""
        self.rfc = rfc

    def __call__(self, x: List[ArithRef]) -> int:
        """Return the output of a random forest."""
        trees = [Tree(t)(x) for t in self.rfc]
        w = [int(t.oob_score * 1000) for t in self.rfc]
        output = _argmax(_sum_weight(trees, w))
        return output


class MUC:
    def __init__(self, rfc: RFC) -> None:
        """Compute MUC based on random forest."""
        self.rf = RandomForest(rfc)
        self.__n_features = rfc.n_features_
        # forest output placeholder
        self.output = None
        self.solution = None

    def muc(self, X: Sequence, y: int, tau: Sequence[float] = None,
            nominal: Dict = None) -> Set[int]:
        """Get the minimal unsatisfiable core of the random forest.

        Args:
          X: Data instance, 1D array.
          y: Ground truth of X.
          tau: Search scope of each feature.
          nominal: Search space of nominal features.

        Returns:
          A set of feature indices, which is the MUC.
        """
        assert len(X) == self.__n_features and isinstance(X[0], (int, float)), \
            f'data instance must be 1D array with size {self.__n_features}'
        if nominal is None:
            nominal = dict()

        # create variables
        variables = [Int(f'x_{i}') if i in self.rf.rfc.nominal_features
                     or i in sum((lst for lst in self.rf.rfc.binary.values()), [])
                     else Real(f'x_{i}') for i in range(self.__n_features)]
        props = [Bool(f'p_{i}') for i in range(self.__n_features)]
        if self.output is None:
            self.output = self.rf(variables)

        # solve
        s = Solver()
        s.set(':core.minimize', True)
        s.add(*_implies(props, _eq_bound(variables, X, tau, nominal)),
              self.output != y)
        if s.check(props) == sat:
            model = s.model()
            self.solution = [float(f'{model[var]}'.replace('?', '')) for var in variables]
        # return a set of feature indices
        return set([int(f'{co}'.split('_')[-1]) for co in s.unsat_core()])

    def predict(self, x: Sequence) -> int:
        return self.rf.rfc(x)
