import json
import os
import pandas as pd
from multiprocessing import Pool
from functools import partial
from typing import Tuple, List


def load_data(model_path: str, test_file: str) -> Tuple[List[List], List, int]:
    """Load test data from file. Change nominal features (str) into indices (int).

    Returns:
      Tuple[X_test, y_test, label_column].
    """
    test_data = pd.read_csv(test_file)
    columns = list(test_data.columns)
    if os.path.exists(os.path.join(model_path, 'settings.json')):
        with open(os.path.join(model_path, 'settings.json')) as f:
            settings = json.load(f)
        label_column = columns.index(settings['output-feature'])
    else:
        label_column = len(columns) - 1
    with open(os.path.join(model_path, 'summary.json')) as f:
        summary = json.load(f)
    classes = summary['output-labels']
    test_data = test_data.values.tolist()

    X_test = [sample[:label_column] + sample[label_column + 1:] for sample in test_data]
    y_test = [classes.index(str(sample[label_column])) for sample in test_data]

    # change nominal features
    with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    label = metadata['features'][label_column]['feature-name']
    dic = {f['name']: f for f in metadata['attributes']}
    i = 0
    for f in metadata['features']:
        if f['attribute-name'] == label:
            continue
        if 'values' in dic[f['attribute-name']]:
            values = dic[f['attribute-name']]['values']
            for j, x in enumerate(X_test):
                X_test[j][i] = values.index(str(x[i]))
        i += 1

    return X_test, y_test, label_column


def _parallel(func, iterable, processes=None, **kwargs):
    pool = Pool(processes)
    func = partial(func, **kwargs)
    res = pool.map(func, iterable)
    pool.close()
    pool.join()
    return list(res)


def _compute_muc(data, muc):
    return muc.muc(data[0], data[1])
