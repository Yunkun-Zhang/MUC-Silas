import os
import argparse
from utils import load_data
from model import RFC, MUC
from analysis import Shapley, AdversarialSample

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model-path', required=True,
                    help='root of your Silas model')
parser.add_argument('-t', '--test-file', required=True,
                    help='path to your test file')
parser.add_argument('-M', type=int, default=None,
                    help='number of subsets sampled for M-Shapley values')
parser.add_argument('-S', action='store_true', help='to compute M-Shapley values')
parser.add_argument('-A', action='store_true', help='to generate adversarial samples')
parser.add_argument('-r', type=int, help='rounding for float values')
args = vars(parser.parse_args())

if __name__ == '__main__':
    model_path, test_file = args['model_path'], args['test_file']
    X_test, y_test, label_column = load_data(model_path, test_file, args['r'])
    rfc = RFC(model_path, X_test, y_test, label_column)
    muc = MUC(rfc)

    # M-Shapley values
    if args['S']:
        print('========== M-Shapley ==========')
        shapley = Shapley(muc, verbose=True)
        if not os.path.exists('muc_save'):
            os.mkdir('muc_save')
        save_file = f'muc_save/{model_path.split("/")[-1]}.json'
        shapley.compute_muc(X_test, y_test, save_file)
        res = dict()
        for c in range(rfc.n_classes_):
            res[c] = shapley.value(c, args['M'])
            print(f'class "{rfc.classes_[c]}":')
            for f in res[c]:
                print(f'\t"{rfc.features_[f]}": {res[c][f]:.6f}')
        print()

    # Adversarial sample
    if args['A']:
        print('========== Adversarial Sample ==========')
        print('WARNING: nominal features should be one-hot encoded.')
        x, y = X_test[0], y_test[0]
        print('Generating for x:  ', x)
        print('Original class:    ', muc.predict(x))
        adv = AdversarialSample(muc, verbose=True)
        opt_sample = adv.opt_adv_sample(
            x, y, num_samples=10000, num_itr=100
        )
        print('Opt sample:        ', opt_sample)
        print('Distance:          ', adv.distance(x, opt_sample))
        print('Adv sample class:  ', muc.predict(opt_sample))
