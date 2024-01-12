from sklearn import preprocessing
from utils.helper import *
from trustscore.trustscore_evaluation import run_clf
import pickle
import warnings
import sys
from trustscore import trustscore
import numpy as np
from sklearn.metrics import matthews_corrcoef
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")

RAND_SEED = 222
np.random.seed(RAND_SEED)

# 5*5交叉验证
n_splits = 5
n_repeats = 5


def calc_performance(label_true, label_pred):
    MCC = matthews_corrcoef(label_true, label_pred)
    return MCC


def load_jit_data(folder_path):
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


def test_model(fname, data, label, clfs, results):

    dpts = trustscore.DPTS()
    all_clf_df = pd.DataFrame()

    for clf in clfs:
        clf_df = pd.DataFrame(columns=['Classifier', 'MCC_before', 'MCC_after'])

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

                model = results[n_repeat][clf][n_split]
                y_preds, confidences = run_clf(model, test_data)
                mcc = calc_performance(test_label, y_preds)
                adjusted_trust_score = dpts.get_score(test_data, y_preds, confidences)

                sorted_indices = np.argsort(adjusted_trust_score)
                remove_count = int(len(sorted_indices) * 0.1)

                filtered_test_data = test_data[sorted_indices[remove_count:]]
                filtered_test_label = test_label[sorted_indices[remove_count:]]

                y_preds_filtered, confidences_filtered = run_clf(model, filtered_test_data)
                mcc_ = calc_performance(filtered_test_label, y_preds_filtered)

                # 将结果添加到DataFrame中
                clf_df = clf_df.append({'Classifier': clf, 'MCC_before': mcc, 'MCC_after': mcc_}, ignore_index=True)

        all_clf_df = pd.concat([all_clf_df, clf_df], axis=0)

    all_clf_df.to_csv('dump/csvs/RQ3/{}.csv'.format(fname), index=False)


if __name__ == '__main__':
    folder_path = 'datasets/'
    data_list, label_list, fname = load_data(folder_path)

    jit_folder_path = 'datasets/NEW-JIT/'
    jit_data_list, jit_label_list, jit_fname = load_jit_data(jit_folder_path)

    data_list.extend(jit_data_list)
    label_list.extend(jit_label_list)
    fname.extend(jit_fname)

    # 采用的分类器方法
    clfs = ['LR', 'SVM', 'NB', 'DT', 'RF', 'GB', 'MLP', 'TabNet']

    for n in range(len(fname)):

        print("*" * 20)
        print('File: ' + fname[n] + '...')
        print("*" * 20)

        data = data_list[n]
        label = label_list[n]

        models_pkfile = open('dump/classifiers/' + fname[n] + '.pickle', 'rb')
        results = pickle.load(models_pkfile)
        test_model(fname[n], data, label, clfs, results)
