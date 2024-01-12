import pandas as pd
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


def save_result(datasets_folder):
    # 获取所有数据集文件
    folder_path = 'datasets/'
    _, _, fname = load_data(folder_path)

    jit_folder_path = 'datasets/NEW-JIT/'
    _, _, jit_fname = load_jit_data(jit_folder_path)

    fname.extend(jit_fname)

    columns = ['MCC_before_LR', 'MCC_after_LR', 'MCC_before_SVM', 'MCC_after_SVM', 'MCC_before_NB', 'MCC_after_NB',
               'MCC_before_DT', 'MCC_after_DT', 'MCC_before_RF', 'MCC_after_RF', 'MCC_before_GB', 'MCC_after_GB',
               'MCC_before_MLP', 'MCC_after_MLP', 'MCC_before_TabNet', 'MCC_after_TabNet']

    all_datasets_df = pd.DataFrame(columns=columns)

    # dataset_files = [os.path.splitext(f)[0] for f in os.listdir(datasets_folder) if f.endswith('.csv')]
    # avg_mcc_before_df = pd.DataFrame()
    # avg_mcc_after_df = pd.DataFrame()

    for dataset_file in fname:
        dataset_path = os.path.join(datasets_folder, dataset_file + '.csv')
        dataset_df = pd.read_csv(dataset_path)
        average_values = dataset_df.groupby('Classifier').mean()

        desired_order = ['LR', 'SVM', 'NB', 'DT', 'RF', 'GB', 'MLP', 'TabNet']
        ordered_averages = average_values.reindex(desired_order)
        flattened_averages = ordered_averages.T.unstack().reset_index(name='Value').drop(columns='level_1')['Value']
        averages_values = flattened_averages.tolist()

        # 将averages_values作为新行添加到DataFrame
        all_datasets_df = all_datasets_df.append(pd.Series(averages_values, index=all_datasets_df.columns), ignore_index=True)

    # 保存DataFrame到CSV文件
    all_datasets_df.to_csv('dump/csvs/RQ3/results_rq3.csv', index=False)
        # dataset_df['MCC_before_avg'] = dataset_df.groupby('Classifier')['MCC_before'].transform('mean')
        # avg_mcc_before_df = pd.concat([avg_mcc_before_df, dataset_df[['Classifier', 'MCC_before_avg']]])
        #
        # dataset_df['MCC_after_avg'] = dataset_df.groupby('Classifier')['MCC_after'].transform('mean')
        # avg_mcc_after_df = pd.concat([avg_mcc_after_df, dataset_df[['Classifier', 'MCC_after_avg']]])

        # Merge DataFrames based on 'Classifier'
    # merged_df = pd.merge(avg_mcc_before_df, avg_mcc_after_df, on=['Classifier'])

    # # Specify the order of classifiers
    # clfs = ['LR', 'SVM', 'NB', 'DT', 'RF', 'GB', 'MLP', 'TabNet']
    #
    # # Reorder columns based on the specified classifier order
    # columns_order = ['Dataset'] + [clf + '_MCC_before' for clf in clfs] + [clf + '_MCC_after' for clf in clfs]
    #
    # # Save the merged DataFrame to a new CSV file
    # merged_df.columns = columns_order
    # merged_df.to_csv(os.path.join(datasets_folder, 'avg_mcc.csv'), index=False)


if __name__ == '__main__':
    datasets_folder = 'dump/csvs/RQ3/'
    save_result(datasets_folder)
