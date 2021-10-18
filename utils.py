import json
import os
import pandas as pd
from typing import Tuple, List


def load_data(model_path: str, test_file: str) -> Tuple[List, List, int]:
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

    return X_test, y_test, label_column
