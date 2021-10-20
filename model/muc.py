from z3 import *
from model.silas import DT, RFC
from typing import List, Set, Sequence


def _abs(x):
    """Return the absolute value of x."""
    return If(x >= 0, x, -x)


def _argmax(lst: List) -> int:
    """Get argmax from a list."""
    m = lst[0]
    j = 0
    for i, x in enumerate(lst):
        j = If(x > m, i, j)
    return j


def _eq_bound(lst, rng, tau=None):
    """Return a Z3 expr whether rng - tau <= lst <= rng + tau."""
    if tau is None:
        return [v1 == v2 for v1, v2 in zip(lst, rng)]
    return [_abs(v1 - v2) <= t for v1, v2, t in zip(lst, rng, tau)]


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
    return Or([x == val for val in lst])


def _implies(p, x):
    """Connect each bool variable to an expression."""
    return [Implies(prop, expr) for prop, expr in zip(p, x)]


class Tree:
    def __init__(self, dt: DT) -> None:
        """Build CNF from a decision tree with only numeric features."""
        # TODO: Deal with nominal features.
        self.dt = dt

    def cnf(self, x: List[ArithRef], out: List[ArithRef]) -> BoolRef:
        """Return Pi(x), Eq 1, 2."""
        disjunctions = []
        t = self.dt
        for rule in t:
            conjuncts = []
            # root and each intermediate node
            for n in rule[1:]:
                p = t.parent[n]
                threshold = t.threshold[p]
                f = t.feature[p]
                if t.children_left[p] == n:
                    if isinstance(threshold, list):
                        conjuncts.append(_in(x[f], threshold))
                    else:
                        conjuncts.append(x[f] <= threshold)
                else:
                    if isinstance(threshold, list):
                        conjuncts.append(Not(_in(x[f], threshold)))
                    else:
                        conjuncts.append(x[f] > threshold)
            # leaf node
            conjuncts.extend(_eq_bound(out, list(t.value[rule[-1]])))
            disjunctions.append(And(conjuncts))
        return Or(disjunctions)


class RandomForest:
    def __init__(self, rfc: RFC) -> None:
        """Build CNF from a random forest."""
        self.rfc = rfc

    def cnf(self, x: List[ArithRef], y: ArithRef) -> BoolRef:
        """Return R(x) and output."""
        outs = [
            [Int(f'out_{i}_{j}') for j in range(self.rfc.n_classes_)]
            for i in range(len(self.rfc.trees_))
        ]
        trees = [Tree(t).cnf(x, outs[i]) for i, t in enumerate(self.rfc.trees_)]
        w = [int(t.oob_score * 10000) for t in self.rfc]
        output = [_argmax(_sum_weight(outs, w)) == y]
        return And(trees + output)


class MUC:
    def __init__(self, rfc: RFC) -> None:
        """Compute MUC based on random forest."""
        self.rf = RandomForest(rfc)
        self.__n_features = rfc.n_features_

    def muc(self, X: Sequence, y: int, tau: Sequence[float] = None) -> Set[int]:
        """Get the minimal unsatisfiable core of the random forest.

        Args:
            X: Data instance, 1D array.
            y: Ground truth of X.
            tau: Search scope of each feature.

        Returns:
            A set of feature indices, which is the MUC.
        """
        assert len(X) == self.__n_features and isinstance(X[0], (int, float)), \
            f'data instance must be 1D array with size {self.__n_features}'

        # create feature variables according to rfc features
        variables = [Int(f'x_{i}') if i in self.rf.rfc.nominal_features
                     else Real(f'x_{i}') for i in range(self.__n_features)]
        # create bool variables to represent MUC
        props = [Bool(f'p_{i}') for i in range(self.__n_features)]
        # create label variable
        pred = Int('y')
        # solve
        s = Solver()
        s.set(':core.minimize', True)
        s.add(self.rf.cnf(variables, pred),
              *_implies(props, _eq_bound(variables, X, tau)),
              pred != y)
        s.check(props)
        # return a set of feature indices
        return set([int(f'{co}'.split('_')[-1]) for co in s.unsat_core()])

    def predict(self, x: Sequence) -> int:
        return self.rf.rfc(x)
