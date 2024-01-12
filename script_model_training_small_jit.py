from sklearn import preprocessing
from utils.helper import *
from src.classifiers_HPO import *
import pickle
import warnings
import sys
import numpy as np
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")


RAND_SEED = 222
np.random.seed(RAND_SEED)


def load_data(folder_path):
    # Initialize lists to store features and labels
    data_list = []
    label_list = []
    fname_list = []

    # Iterate over each CSV file in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # Construct the full path to the CSV file
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file using pandas
            df = pd.read_csv(file_path, index_col=None, header=0,  sep='[:,;]', engine='python')
            # pd.read_csv(file_path)
            df.dropna(inplace=True)

            # Extract features (excluding first two columns and last column)
            features = df.values[:, :-1]

            # Extract labels (last column)
            labels = df.values[:, -1].astype('int')

            # Append features and labels to the lists
            data_list.append(features)
            label_list.append(labels)
            fname_list.append(filename[:-4])

    return data_list, label_list, fname_list


def main_experiment(data, label, n_splits, n_repeats, clfs, metric):

    results = {}

    for n_repeat in range(n_repeats):

        results[n_repeat] = {}

        # 通过5*5交叉验证的方式划分数据集
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=n_repeat)

        for clf in clfs:
            results[n_repeat][clf] = []

        results[n_repeat]['train_inds'] = []
        results[n_repeat]['test_inds'] = []

        # k-fold交叉验证
        for train_ind, test_ind in skf.split(data, label):
            train_data, test_data = data[train_ind], data[test_ind]
            train_label, test_label = label[train_ind], label[test_ind]

            results[n_repeat]['train_inds'].append(train_ind)
            results[n_repeat]['test_inds'].append(test_ind)

            # 数据归一化处理
            scaler = preprocessing.StandardScaler().fit(train_data)
            train_data = scaler.transform(train_data)

            # 遍历分类算法
            for clf in clfs:

                print("repeat: " + str(1+n_repeat) + "  "+clf)

                if clf == 'KNN':
                    model = KNN_Opt(train_data, train_label, metrics=metric, opt_algo='TPE')

                elif clf == 'NB':
                    model = NB_Opt(train_data, train_label, metrics=metric, opt_algo='TPE')

                elif clf == 'LR':
                    model = LR_Opt(train_data, train_label, metrics=metric, opt_algo='TPE')

                elif clf == 'MLP':
                    model = MLP_Opt(train_data, train_label, metrics=metric, opt_algo='TPE')

                elif clf == 'SVM':
                    model = SVM_Opt(train_data, train_label, metrics=metric, opt_algo='TPE')

                elif clf == 'DT':
                    model = DT_Opt(train_data, train_label, metrics=metric, opt_algo='TPE')

                elif clf == 'RF':
                    model = RF_Opt(train_data, train_label, metrics=metric, opt_algo='TPE')

                elif clf == 'GB':
                    model = GB_Opt(train_data, train_label, metrics=metric, opt_algo='TPE')

                results[n_repeat][clf].append(model)

    return results


if __name__ == '__main__':
    folder_path = 'datasets/NEW-JIT/'
    data_list, label_list, fname = load_data(folder_path)

    # 5*5交叉验证
    n_splits = 5
    n_repeats = 5

    # 采用的分类器方法
    clfs = ['KNN', 'LR', 'SVM', 'NB', 'DT', 'RF', 'GB', 'MLP']

    # 最终的结果包括各个数据集下各个分类器的各种分类性能指标值
    metric = "MCC"

    for n in range(12):
        print("*" * 20)
        print('File: ' + fname[n] + '...')
        print("*" * 20)

        data = data_list[n]
        label = label_list[n]

        pkfile = open('dump/classifiers/' + fname[n] + '.pickle', 'wb')

        results = main_experiment(data, label, n_splits, n_repeats, clfs, metric)

        pickle.dump(results, pkfile)
