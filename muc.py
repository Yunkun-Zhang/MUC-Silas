from z3 import *
from silas import DT, RFC
from typing import List, Set


def _argmax(lst: List) -> int:
    """Get argmax from a list."""
    m = lst[0]
    j = 0
    for i, x in enumerate(lst):
        j = If(x > m, i, j)
    return j


def _eq(list1: List, list2: List):
    """Return a Z3 expr whether list1 == list2."""
    return [v1 == v2 for v1, v2 in zip(list1, list2)]


def _sum(lists: List[List[ArithRef]], weights: List[int] = None):
    """Add up some Real lists with weight."""
    res = [0] * len(lists[0])
    if weights is None:
        weights = [1] * len(lists)
    for ind, lst in enumerate(lists):
        for i, val in enumerate(lst):
            res[i] = res[i] + val * weights[ind]
    return res


def _implies(p: List[BoolRef], x: List[ExprRef]):
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
                if t.children_left[p] == n:
                    conjuncts.append(x[t.feature[p]] <= t.threshold[p])
                else:
                    conjuncts.append(x[t.feature[p]] > t.threshold[p])
            # leaf node
            conjuncts.append(And(_eq(out, list(t.value[rule[-1]]))))
            disjunctions.append(And(conjuncts))
        return Or(disjunctions)


class RandomForest:
    def __init__(self, rfc: RFC) -> None:
        """Build CNF from a random forest."""
        self.rfc = rfc

    def cnf(self, x: List[ArithRef], y: ArithRef) -> BoolRef:
        outs = [
            [Int(f'out_{i}_{j}') for j in range(self.rfc.n_classes_)]
            for i in range(len(self.rfc.trees_))
        ]
        trees = [Tree(t).cnf(x, outs[i]) for i, t in enumerate(self.rfc.trees_)]
        # does not consider tree weights cause it will be very slow
        w = [int(t.oob_score * 10000) for t in self.rfc]
        output = [_argmax(_sum(outs, w)) == y]
        return And(trees + output)


class MUC:
    def __init__(self, rfc: RFC) -> None:
        self.rf = RandomForest(rfc)

    def muc(self, X: List, y: int) -> Set[int]:
        """Get the minimal unsatisfiable core of the random forest.

        Args:
            X: Data instance, 1D array.
            y: Ground truth of X.

        Returns:
            A set of feature indices, which is the MUC.
        """
        variables = [Real(f'x_{i}') for i in range(self.rf.rfc.n_features_)]
        props = [Bool(f'p_{i}') for i in range(self.rf.rfc.n_features_)]
        pred = Int('y')
        s = Solver()
        s.set(':core.minimize', True)
        s.add(self.rf.cnf(variables, pred),
              *_implies(props, _eq(variables, X)),
              pred != y)
        s.check(props)
        return set([int(f'{co}'.split('_')[-1]) for co in s.unsat_core()])
