import os
import argparse
from utils import load_data
from model import RFC, MUC
from analysis import Shapley, AdversarialSample

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model-path', default='SilasModel/diabetes',
                    help='root of your Silas model')
parser.add_argument('-t', '--test-file', default='data/diabetes.csv',
                    help='path to your test file')
parser.add_argument('-M', type=int, default=256,
                    help='number of subsets sampled for M-Shapley values')
args = vars(parser.parse_args())

if __name__ == '__main__':
    X_test, y_test, label_column = load_data(args['model_path'], args['test_file'])
    rfc = RFC(args['model_path'], X_test, y_test, label_column)
    muc = MUC(rfc)

    # M-Shapley values
    print('========== M-Shapley ==========')
    shapley = Shapley(muc)
    if not os.path.exists('muc_save'):
        os.mkdir('muc_save')
    shapley.compute_muc(X_test, y_test, 'muc_save/diabetes.json')
    res = dict()
    for c in range(rfc.n_classes_):
        res[c] = shapley.value(c, args['M'])
        print(f'class "{rfc.classes_[c]}":')
        for f in res[c]:
            print(f'\t"{rfc.features_[f]}": {res[c][f]:.6f}')
    print()

    # Adversarial sample
    print('========== Adversarial Sample ==========')
    x, y = X_test[0], y_test[0]
    print('Generating for x:  ', x)
    print('Original class:    ', muc.predict(x))
    adv = AdversarialSample(muc, verbose=False)
    opt_sample = adv.opt_adv_sample(x, y, num_samples=50000, num_itr=50)
    print('Opt sample:        ', opt_sample)
    print('Distance:          ', adv.distance(x, opt_sample))
    print('Adv sample class:  ', muc.predict(opt_sample))
