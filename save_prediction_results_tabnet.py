from sklearn import preprocessing
from utils.helper import *
from trustscore.trustscore_evaluation import run_clf
import pickle
import warnings
import sys
from trustscore import trustscore
import numpy as np
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")


RAND_SEED = 222
np.random.seed(RAND_SEED)

# 5*5交叉验证
n_splits = 5
n_repeats = 5


def test_model(data, label, clfs, results):
    ndata = data.shape[0]
    data_splits, test_predictions, test_confidences, test_misclfs, test_dpts = {}, {}, {}, {}, {}
    dpts = trustscore.DPTS()

    for n_repeat in range(n_repeats):
        data_splits[n_repeat] = {}
        data_splits[n_repeat]['train_inds'] = results[n_repeat]['train_inds']
        data_splits[n_repeat]['test_inds'] = results[n_repeat]['test_inds']

    for clf in clfs:
        test_predictions[clf] = np.zeros((ndata, n_repeats))
        test_confidences[clf] = np.zeros((ndata, n_repeats))
        test_misclfs[clf] = np.zeros((ndata, n_repeats))
        test_dpts[clf] = np.zeros((ndata, n_repeats))

    for n_repeat in range(n_repeats):
        for n_split in range(n_splits):

            # k-fold交叉验证
            train_ind, test_ind = results[n_repeat]['train_inds'][n_split], results[n_repeat]['test_inds'][n_split]

            train_data, test_data = data[train_ind], data[test_ind]
            train_label, test_label = label[train_ind], label[test_ind]

            # 数据归一化处理
            scaler = preprocessing.StandardScaler().fit(train_data)
            train_data = scaler.transform(train_data)
            test_data = scaler.transform(test_data)

            dpts.fit(train_data, train_label)

            # 遍历分类算法
            for clf in clfs:
                model = results[n_repeat][clf][n_split]
                y_preds, confidences = run_clf(model, test_data)
                dpts_results = dpts.get_score(test_data, y_preds, confidences)

                for i in range(y_preds.shape[0]):

                    if y_preds[i] != test_label[i]:
                        test_misclfs[clf][test_ind[i]][n_repeat] = 1

                    test_predictions[clf][test_ind[i]][n_repeat] = y_preds[i]
                    test_confidences[clf][test_ind[i]][n_repeat] = confidences[i]
                    test_dpts[clf][test_ind[i]][n_repeat] = dpts_results[i]

    return data_splits, test_predictions, test_confidences, test_dpts, test_misclfs


if __name__ == '__main__':
    folder_path = 'datasets/'
    data_list, label_list, fname = load_data(folder_path)
    files = os.listdir(folder_path)

    # 采用的分类器方法
    clfs = ['TabNet']

    for n in [0]:

        print("*" * 20)
        print('File: ' + fname[n] + '...')
        print("*" * 20)

        data = data_list[n]
        label = label_list[n]

        models_pkfile = open('dump/classifiers/' + fname[n] + '.pickle', 'rb')
        results = pickle.load(models_pkfile)

        tests_pkfile = open('dump/test_predictions/' + fname[n] + '.pickle', 'wb')
        data_splits, test_predictions, test_confidences, test_dpts, test_misclfs = test_model(data, label, clfs, results)

        pickle.dump(data_splits, tests_pkfile)
        pickle.dump(test_predictions, tests_pkfile)
        pickle.dump(test_confidences, tests_pkfile)
        pickle.dump(test_dpts, tests_pkfile)
        pickle.dump(test_misclfs, tests_pkfile)

