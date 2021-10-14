import json
import os
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Tuple

__version__ = '0.8.7'


def load_data(model_path: str, test_file: str) -> Tuple[List, List]:
    """Load test data from file.

    Returns:
      Tuple[X_test, y_test].
    """
    test_data = pd.read_csv(test_file)
    columns = list(test_data.columns)
    if os.path.exists(os.path.join(model_path, 'settings.json')):
        with open(os.path.join(model_path, 'settings.json')) as f:
            settings = json.load(f)
        label_column = columns.index(settings['output-feature'])
    else:
        label_column = len(columns) - 1
    test_data = test_data.values.tolist()

    X_test = [sample[:label_column] + sample[label_column + 1:] for sample in test_data]
    y_test = [sample[label_column] for sample in test_data]

    return X_test, y_test


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
                else:
                    self.children_left[number], value1 = dfs(d['right'], number)
                    self.children_right[number], value2 = dfs(d['left'], number)
                value = value1 + value2
                self.ig[number] = d['weight']
                self.threshold[number] = d['threshold'] if 'threshold' in d else d['partition']
            rule.pop()
            self.value[number] = value
            return number, value

        dfs(dic, -1)

    def __getitem__(self, item: int) -> List[int]:
        return self.rules[item]


class RFC:
    """Random forest classifier."""

    def __init__(self,
                 model_path='',
                 test_file='test.csv',
                 pred_file='predictions.csv',
                 label_column: int = None) -> None:
        """Build a random forest classifier.

        Args:
          model_path: Path to Silas model.
          test_file: Path of test data file.
          pred_file: Path of Silas prediction file.
          label_column: Column index.
        """
        self.model_path = model_path

        with open(os.path.join(model_path, 'summary.json'), 'r') as f:
            summary = json.load(f)
        with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        if label_column is None:
            label_column = len(metadata['features']) - 1
        label = metadata['features'][label_column]['feature-name']

        # for nominal feature method 'satisfy'
        dic = {f['name']: f for f in metadata['attributes']}
        self.nominal_features = []
        for f in metadata['features']:
            if f['attribute-name'] == label:
                continue
            if 'values' in dic[f['attribute-name']]:
                self.nominal_features.append(dic[f['attribute-name']]['values'])
            else:
                self.nominal_features.append(None)

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
        self.trees_oob_scores = []

        self._set_oob_scores(summary['trees'])  # read tree oob scores
        self._build_trees(summary['trees'])  # build decision trees

        # record test data
        self.X_test, self.y_test = load_data(model_path, test_file)
        self.y_test = [self.classes_.index(y) for y in self.y_test]
        # record prediction
        self.prob = np.array(pd.read_csv(pred_file).values.tolist())
        self.pred = np.array([self.classes_[c] for c in np.argmax(self.prob, axis=-1)])

    def __getitem__(self, item: int) -> DT:
        return self.trees_[item]

    def _set_oob_scores(self, trees_dic):
        for tree in trees_dic:
            # in Silas v0.8.7 this is the oob score
            self.trees_oob_scores.append(tree['weight'])

    def _build_trees(self, trees_dic):
        for tree in trees_dic:
            with open(os.path.join(self.model_path, tree['path']), 'r') as f:
                d = json.load(f)
            dt = DT()
            dt.initialize(tree['weight'], d, self.feature_indices)
            self.trees_.append(dt)

    def predict(self, x):
        """Classify data sample x from test data."""
        try:
            ind = self.X_test.index(x)
        except ValueError:
            print('Not a test data!')
            return -1
        return self.pred[ind]
