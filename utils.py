import json
import os
import pandas as pd
from multiprocessing import Pool
from functools import partial
from typing import Tuple, List


def nominal_to_binary(csv, save_file):
    """Convert nominal features into one-hot encoding."""
    df = pd.read_csv(csv)
    columns = list(df.columns)
    data = df.values.tolist()

    try:
        with open(f'clean-metadata.json') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        with open(f'metadata.json') as f:
            metadata = json.load(f)

    s = 0
    new_columns = []
    for i, col in enumerate(columns[:-1]):
        attr = metadata['attributes'][i + s].copy()
        if attr['type'] == 'nominal':
            values = attr['values'].copy()
            for j, x in enumerate(data):
                ind = values.index(str(x[i + s]))
                data[j].pop(i + s)
                for k in range(len(values)):
                    data[j].insert(i + s + k, 0)
                data[j][i + s + ind] = 1
            s += len(values) - 1
        else:
            new_columns.append(col)
    new_columns.append(columns[-1])

    df = pd.DataFrame(data, columns=new_columns)
    df.to_csv(save_file, index=False)


def load_data(model_path: str, test_file: str, r: int = 6) -> Tuple[List[List], List, int]:
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
    y_test = [classes.index(_label_float_to_int(sample[label_column])) for sample in test_data]

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
                temp = str(round(float(x[i]), r))
                X_test[j][i] = values.index(temp)
        i += 1

    return X_test, y_test, label_column


def _label_float_to_int(label):
    try:
        label = float(label)
    except ValueError:
        pass
    if isinstance(label, int) or (isinstance(label, float) and label.is_integer()):
        label = str(int(label))
    return label


def _parallel(func, iterable, processes=None, **kwargs):
    pool = Pool(processes)
    func = partial(func, **kwargs)
    res = pool.map(func, iterable)
    pool.close()
    pool.join()
    return list(res)


def _compute_muc(data, muc):
    return muc.muc(data[0], data[1])


def test_func():
    print('Hello, World!')
