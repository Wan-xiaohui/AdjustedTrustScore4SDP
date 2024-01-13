from utils.helper import *
import warnings
import sys
import numpy as np
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")


RAND_SEED = 222
np.random.seed(RAND_SEED)


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


def save_result(is_correct=True):
    folder_path = 'datasets/'
    _, _, fname = load_data(folder_path)

    jit_folder_path = 'datasets/NEW-JIT/'
    _, _, jit_fname = load_jit_data(jit_folder_path)

    fname.extend(jit_fname)

    # 采用的分类器方法
    clfs = ['LR', 'SVM', 'NB', 'DT', 'RF', 'GB', 'MLP', 'TabNet']

    if is_correct:
        for clf_index in range(8):
            clf = clfs[clf_index]
            final_results = pd.DataFrame()

            for n in range(len(fname)):
                print("*" * 20)
                print('File: ' + fname[n] + '...')
                print("*" * 20)

                result = pd.read_csv('dump/csvs/RQ1/Correct/' + fname[n] + ".csv")
                result.insert(0, 'dataset', os.path.splitext(fname[n])[0])
                final_results = pd.concat([final_results, result.iloc[clf_index, -3:]], axis=1)

            final_results = final_results.T
            final_results.insert(0, 'dataset', [fname[i] for i in range(len(fname))])
            final_results.to_csv('dump/csvs/RQ1/results_correct_{}.csv'.format(clf), index=False)

    else:
        for clf_index in range(8):
            clf = clfs[clf_index]
            final_results = pd.DataFrame()

            for n in range(len(fname)):
                print("*" * 20)
                print('File: ' + fname[n] + '...')
                print("*" * 20)

                result = pd.read_csv('dump/csvs/RQ1/Incorrect/' + fname[n] + ".csv")
                result.insert(0, 'dataset', os.path.splitext(fname[n])[0])
                final_results = pd.concat([final_results, result.iloc[clf_index, -3:]], axis=1)

            final_results = final_results.T
            final_results.insert(0, 'dataset', [fname[i] for i in range(len(fname))])
            final_results.to_csv('dump/csvs/RQ1/results_incorrect_{}.csv'.format(clf), index=False)


def clf_analysis(is_correct=True):
    folder_path = 'datasets/'
    _, _, fname = load_data(folder_path)

    jit_folder_path = 'datasets/NEW-JIT/'
    _, _, jit_fname = load_jit_data(jit_folder_path)

    fname.extend(jit_fname)

    # 采用的分类器方法
    clfs = ['LR', 'SVM', 'NB', 'DT', 'RF', 'GB', 'MLP', 'TabNet']

    if is_correct:
        for clf_index in range(8):
            clf = clfs[clf_index]
            final_results = pd.DataFrame()

            for n in range(len(fname)):
                print("*" * 20)
                print('File: ' + fname[n] + '...')
                print("*" * 20)

                result = pd.read_csv('dump/csvs/RQ1/Correct/' + fname[n] + ".csv")
                result.insert(0, 'dataset', os.path.splitext(fname[n])[0])
                final_results = pd.concat([final_results, result.iloc[clf_index, -3:]], axis=1)

            final_results = final_results.T
            final_results.insert(0, 'dataset', [fname[i] for i in range(len(fname))])
            final_results.to_csv('dump/csvs/RQ1/results_correct_{}.csv'.format(clf), index=False)

    else:
        for clf_index in range(8):
            clf = clfs[clf_index]
            final_results = pd.DataFrame()

            for n in range(len(fname)):
                print("*" * 20)
                print('File: ' + fname[n] + '...')
                print("*" * 20)

                result = pd.read_csv('dump/csvs/RQ1/Incorrect/' + fname[n] + ".csv")
                result.insert(0, 'dataset', os.path.splitext(fname[n])[0])
                final_results = pd.concat([final_results, result.iloc[clf_index, -3:]], axis=1)

            final_results = final_results.T
            final_results.insert(0, 'dataset', [fname[i] for i in range(len(fname))])
            final_results.to_csv('dump/csvs/RQ1/results_incorrect_{}.csv'.format(clf), index=False)


def total_analysis(is_correct=True):
    total_results = pd.DataFrame()
    clfs = ['LR', 'SVM', 'NB', 'DT', 'RF', 'GB', 'MLP', 'TabNet']

    if is_correct:
        for clf_index in range(8):
            clf = clfs[clf_index]

            # Read the CSV file for the current classifier
            current_classifier_data = pd.read_csv('dump/csvs/RQ1/results_correct_{}.csv'.format(clf))

            # If it's the first classifier (LR), keep the dataset column
            if clf_index == 0:
                total_results = current_classifier_data

            else:
                # For subsequent classifiers, drop the dataset column and append the remaining three columns
                total_results = pd.concat([total_results, current_classifier_data.iloc[:, 1:]], axis=1)

        # Save the total_results DataFrame to a CSV file
        total_results.to_csv('dump/csvs/RQ1/total_results_correct.csv', index=False)

    else:
        for clf_index in range(8):
            clf = clfs[clf_index]

            # Read the CSV file for the current classifier
            current_classifier_data = pd.read_csv('dump/csvs/RQ1/results_incorrect_{}.csv'.format(clf))

            # If it's the first classifier (LR), keep the dataset column
            if clf_index == 0:
                total_results = current_classifier_data

            else:
                # For subsequent classifiers, drop the dataset column and append the remaining three columns
                total_results = pd.concat([total_results, current_classifier_data.iloc[:, 1:]], axis=1)

            # Save the total_results DataFrame to a CSV file
        total_results.to_csv('dump/csvs/RQ1/total_results_incorrect.csv', index=False)


if __name__ == '__main__':

    save_result(is_correct=True)
    clf_analysis(is_correct=True)
    total_analysis(is_correct=True)

    clf_analysis(is_correct=False)
    total_analysis(is_correct=False)
