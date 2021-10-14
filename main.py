import os
import argparse
from silas import RFC
from muc import MUC
from analysis import Shapley

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model-path', default='model/diabetes')
parser.add_argument('-t', '--test-file', default='data/diabetes.csv')
parser.add_argument('-p', '--prediction-file', default='data/pred_diabetes.csv')
parser.add_argument('-M', type=int, default=256)
args = vars(parser.parse_args())

if __name__ == '__main__':
    rfc = RFC(args['model_path'], args['test_file'], args['prediction_file'])
    muc = MUC(rfc)
    shapley = Shapley(muc)
    if not os.path.exists('muc_save'):
        os.mkdir('muc_save')
    shapley.compute_muc('muc_save/diabetes.json')
    res = dict()
    for c in range(rfc.n_classes_):
        res[c] = shapley.value(c, args['M'])
    print(res)
