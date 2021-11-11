import json
import os
import numpy as np
from typing import Sequence, List, Dict

__version__ = '0.8.7'


class DT:
    """Decision tree."""

    def __init__(self):
        self.oob_score = 0
        self.count = 0              # number of nodes
        self.feature = []           # feature indices at each node
        self.rules = []             # leaf paths
        self.children_left = []
        self.children_right = []
        self.parent = []
        self.value = []             # number of samples at each node
        self.ig = []                # information gain at each node
        self.threshold = []         # feature threshold or partition at each node

    def initialize(self, weight: float, dic: Dict, index: List[int]) -> None:
        """Initialize a decision tree with Silas-generated .json file."""
        self.__init__()
        self.oob_score = weight
        rule = []

        def dfs(d, parent):
            number = self.count
            self.count += 1
            rule.append(number)
            self.feature.append(-1)
            self.children_left.append(-1)
            self.children_right.append(-1)
            self.parent.append(parent)
            self.value.append([])
            self.ig.append(0)
            self.threshold.append(None)
            if 'aggregate' in d:
                self.rules.append(rule[:])
                value = np.array(d['aggregate'])
                prob = value / value.sum()
                # calculate entropy
                entropy = 0
                for p in prob:
                    if p != 0:
                        entropy -= p * np.log2(p)
                self.ig[number] = entropy
            else:
                self.feature[number] = index[d['featureIndex']]
                if 'threshold' in d:
                    self.children_left[number], value1 = dfs(d['left'], number)
                    self.children_right[number], value2 = dfs(d['right'], number)
                    self.threshold[number] = d['threshold']
                else:
                    self.children_left[number], value1 = dfs(d['right'], number)
                    self.children_right[number], value2 = dfs(d['left'], number)
                    self.threshold[number] = [
                        i for i, v in enumerate(d['partition']) if v is True
                    ]
                value = value1 + value2
                self.ig[number] = d['weight']
            rule.pop()
            self.value[number] = value
            return number, value

        dfs(dic, -1)

    def __call__(self, x: Sequence) -> np.ndarray:
        node = 0
        while self.children_left[node] != -1:
            f = self.feature[node]
            t = self.threshold[node]
            if isinstance(t, list):
                if int(x[f]) in t:
                    node = self.children_left[node]
                else:
                    node = self.children_right[node]
            else:
                if x[f] <= t:
                    node = self.children_left[node]
                else:
                    node = self.children_right[node]
        return self.value[node]

    def __getitem__(self, item: int) -> List[int]:
        return self.rules[item]


class RFC:
    """Random forest classifier."""

    def __init__(self, model_path='',
                 X_test: List[Sequence] = None,
                 y_test: List = None,
                 label_column: int = None) -> None:
        """Build a random forest classifier.

        Args:
          model_path: Path to Silas SilasModel.
          X_test: Test data instances.
          y_test: Ground truth of X_test.
          label_column: Column index.
        """
        self.model_path = model_path
        self.X_test = X_test
        self.y_test = y_test

        with open(os.path.join(model_path, 'summary.json'), 'r') as f:
            summary = json.load(f)
        with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        if label_column is None:
            label_column = len(metadata['features']) - 1
        label = metadata['features'][label_column]['feature-name']

        # nominal feature indices
        dic = {f['name']: f for f in metadata['attributes']}
        self.nominal_features = []
        i = 0
        for f in metadata['features']:
            if f['attribute-name'] == label:
                continue
            if 'values' in dic[f['attribute-name']]:
                self.nominal_features.append(i)
            i += 1

        # converted nominal features (binary) indices
        self.binary = dict()
        temp = []
        name = metadata['attributes'][0]['name'].split('_')
        name = '_'.join(name[:-1] if name[-1].isdigit() else name)
        for i, attr in enumerate(metadata['attributes']):
            new_name = attr['name'].split('_')
            if name != '' and '_'.join(new_name[:-1]) == name:
                temp.append(i)
            else:
                if temp:
                    self.binary[name] = temp[:]
                    temp = []
                name = '_'.join(new_name[:-1] if new_name[-1].isdigit() else new_name)
                if new_name[-1].isdigit():
                    temp.append(i)
        if temp:
            self.binary[name] = temp[:]

        # features are ordered according to metadata
        features = [a['feature-name'] for a in metadata['features']]
        label_column = features.index(label)
        self.features_ = features[:label_column] + features[label_column + 1:]
        self.output_feature_ = label

        # revise featureIndex in tree nodes
        self.feature_indices = [self.features_.index(f) for f in summary['template']]

        self.n_estimators = summary['size']
        self.classes_ = summary['output-labels']
        self.n_classes_ = len(self.classes_)
        self.n_features_ = len(self.features_)
        self.n_outputs_ = 1
        self.trees_ = []

        self._build_trees(summary['trees'])  # build decision trees

    def __getitem__(self, item: int) -> DT:
        return self.trees_[item]

    def _build_trees(self, trees_dic):
        for tree in trees_dic:
            with open(os.path.join(self.model_path, tree['path']), 'r') as f:
                d = json.load(f)
            dt = DT()
            dt.initialize(tree['weight'], d, self.feature_indices)
            self.trees_.append(dt)

    def __call__(self, x: Sequence) -> int:
        """Classify data sample x from test data."""
        vote = np.zeros(self.n_classes_)
        for t in self:
            vote += t(x) * t.oob_score
        return np.argmax(vote)
